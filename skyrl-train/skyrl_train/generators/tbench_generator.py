import asyncio
import copy
from uuid import uuid4
import skyrl_gym
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm
import time
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInput, ConversationType
from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
import threading
import terminal_bench
from terminal_bench import Harness
from terminal_bench.agents import AgentName
from pathlib import Path
from datetime import datetime
import logging
from skyrl_train.inference_engines.launch_inference_engine_http_server import (
    serve,
    wait_for_server_ready,
    shutdown_server,
    handle_chat_completion,
)
from transformers import AutoTokenizer
from pathlib import Path
from sandbox.models.task.id import GitTaskId, LocalTaskId
from sandbox.models.agent.name import AgentName
from sandbox.trial.trial import Trial, TrialEvent
import os
import hashlib
from sandbox.models.trial.config import TrialConfig, EnvironmentConfig, AgentConfig, LocalTaskConfig
from sandbox.models.environment_type import EnvironmentType
import aiohttp
import asyncio
from aiohttp_socks import ProxyConnector

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TP_SIZE = 1
SERVER_PORT = 8000
SERVER_HOST = "127.0.0.1"

# Add this to your code where the environment variables are being printed
import subprocess
import socket
def configure_daytona_for_socks():
    print("=== Configuring Daytona for SOCKS proxy ===")
    
    # Create a custom connector for SOCKS proxy
    connector = ProxyConnector.from_url('socks5://localhost:7003')
    
    # Set longer timeouts for aiohttp
    timeout = aiohttp.ClientTimeout(
        total=1800,  # 30 minutes total
        connect=300,  # 5 minutes to connect
        sock_read=900,  # 15 minutes for socket read
        sock_connect=300  # 5 minutes for socket connect
    )
    
    # Configure aiohttp session
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        trust_env=True  # Use environment proxy settings
    )
    
    return session



def test_tunnel_connectivity():
    print("=== Testing SSH Tunnel Connectivity ===")
    
    # Test if port 7003 is listening
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 7003))
        sock.close()
        if result == 0:
            print("Port 7003 is accessible")
        else:
            print(f"Port 7003 is NOT accessible (error code: {result})")
    except Exception as e:
        print(f"Socket test failed: {e}")
    
    # Test SSH tunnel process
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        ssh_processes = [line for line in result.stdout.split('\n') if 'ssh' in line and '7003' in line]
        if ssh_processes:
            print("SSH tunnel processes found:")
            for proc in ssh_processes:
                print(f"  {proc}")
        else:
            print("No SSH tunnel processes found")
    except Exception as e:
        print(f"Process check failed: {e}")
    
    # Test actual connectivity through tunnel
    try:
        import requests
        import urllib3
        urllib3.disable_warnings()
        
        proxies = {'http': 'socks5://localhost:7003', 'https': 'socks5://localhost:7003'}
        response = requests.get('https://httpbin.org/ip', proxies=proxies, timeout=10, verify=False)
        print(f"Tunnel connectivity test: SUCCESS - {response.json()}")
    except Exception as e:
        print(f"Tunnel connectivity test: FAILED - {e}")
    
    print("=== End Tunnel Connectivity Test ===")


class TBenchGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
        """
        # Call parent constructor first
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name)
        

        self.http_server_inference_engine_client_host = generator_cfg.get(
            "http_server_inference_engine_client_host", "127.0.0.1"
        )
        self.http_server_inference_engine_client_port = generator_cfg.get(
            "http_server_inference_engine_client_port", 8000
        )
        self.base_url = f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        # [Marianna] set trial dir as environment var for testing (permission denied)
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer

    async def tbench_agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: List[Dict[str, Any]],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[int], float, str, List[int], List[int]]:
        """
        Multi-turn generation loop that executes a single trajectory.

        Args:
            prompt: ConversationType
            env_extras: List[Dict[str, Any]]
            max_tokens: int
            max_input_length: int
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            response_ids: List[int]
            reward: float
            stop_reason: str
            loss_mask: List[int]
            prompt_token_ids: List[int]
        """        
        trials_dir = self.generator_cfg.get("trial_runs_dir")
        import os
        print("=== Environment Variables in Ray Remote Context ===")
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 
                    'DAYTONA_API_KEY', 'DAYTONA_TIMEOUT', 'AIOHTTP_CLIENT_TIMEOUT']
        for var in proxy_vars:
            print(f"{var}: {os.environ.get(var, 'NOT SET')}")
        print("=== End Environment Variables ===")


        # Call this in your Ray remote function
        test_tunnel_connectivity()
        # Use this before creating Daytona client
        session = configure_daytona_for_socks()
        if self.generator_cfg.get("agent_name") == "terminus":
            self.trial_config = TrialConfig(
                task=LocalTaskConfig(id=LocalTaskId(path=f"{self.generator_cfg.get('sandboxes_dir')}/examples/tasks/hello-world")),
                trials_dir=Path(trials_dir),
                environment=EnvironmentConfig(type=EnvironmentType.DAYTONA),
                agent=AgentConfig(
                    name=AgentName.TERMINUS_2.value,
                    model_name=f"hosted_vllm/{MODEL}",
                    kwargs={"api_base": f"{self.base_url}/v1", "key": "fake_key"},
                )
            )
        elif self.generator_cfg.get("agent_name") == "oracle":
            self.trial_config = TrialConfig(
                task=LocalTaskConfig(id=LocalTaskId(path=f"{self.generator_cfg.get('sandboxes_dir')}/examples/tasks/hello-world")),
                trials_dir=Path(trials_dir),
                environment=EnvironmentConfig(type=EnvironmentType.DAYTONA),
                agent=AgentConfig(
                    name=AgentName.ORACLE,
                    model_name=self.model_name,
                )
            )
        else:
            raise ValueError(f"Invalid agent name: {self.generator_cfg.get('agent_name')}")
        
        trial = Trial(self.trial_config)
        # Run the trial
        while True:
            results = await trial.run()
            reward = results.verifier_result.rewards
            chat_history = results.agent_result.all_messages
            if len(chat_history) > 0:
                break
        
        # Use the first message as the prompt
        prompt = [chat_history[0]]
        initial_input_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,  # Always add generation prompt for multi-turn
            tokenize=True,
        )
        initial_prompt_length = len(initial_input_ids)
        
        # Process response messages (everything after the first message)
        response_messages = chat_history[1:]
        
        response_ids = []
        loss_mask = []

        for message in response_messages:
            # Apply chat template and tokenize each message
            msg_encoding = self.tokenizer.apply_chat_template(
                [message],
                add_generation_prompt=False,
                tokenize=True
            )
            
            # Extend response_ids with the tokens
            response_ids.extend(msg_encoding)
            
            # Extend loss_mask: 0s for user, 1s for assistant
            if message["role"] == "user":
                loss_mask.extend([0] * len(msg_encoding))
            else:  # assistant
                loss_mask.extend([1] * len(msg_encoding))
        # Extract prompt ids
        prompt_ids = initial_input_ids
        
        # Calculate maximum response tokens allowed
        if hasattr(self, 'max_turns') and self.max_turns > 1:
            max_response_tokens = max_tokens + max_input_length - initial_prompt_length
        else:
            max_response_tokens = max_tokens
        
        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"
        
        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        return response_ids, reward, stop_reason, loss_mask, prompt_ids


    async def generate_batched(
        self,
        prompts: List[ConversationType],
        env_classes: List[str],
        env_extras: List[Dict[str, Any]],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> GeneratorOutput:
        """
        Single-turn batched generation (can use the synchronous offline engine)

        Args:
            prompts: List[ConversationType]
            env_classes: List[str]
            env_extras: List[Dict[str, Any]]
            max_tokens: int
            max_input_length: int --> Currently unused as we assume batched is used only for single-turn.
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            GeneratorOutput
        """
        tasks = []

        for i in range(len(prompts)):
            tasks.append(
                self.tbench_agent_loop(
                    "Hello, how are you?",
                    "hello-world",
                    [],
                    1024,
                    1024,
                    sampling_params=sampling_params,
                )
            )
 
        all_outputs = await asyncio.gather(*tasks)

        responses = [output[0] for output in all_outputs]
        rewards = [output[1] for output in all_outputs]
        stop_reasons = [output[2] for output in all_outputs]
        loss_masks = [output[3] for output in all_outputs]
        prompt_token_ids = [output[4] for output in all_outputs]
        rollout_metrics = self._rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output