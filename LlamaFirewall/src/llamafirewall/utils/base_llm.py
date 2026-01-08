# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
import os
from typing import Any, Dict, Generic, List, Literal, Optional, Type, TypeVar

import requests
from openai import OpenAI
from pydantic import BaseModel, ValidationError

logger: logging.Logger = logging.getLogger(__name__)

OutputSchemaT = TypeVar("OutputSchemaT", bound=BaseModel)

import xgrammar as xgr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class LLMClient(Generic[OutputSchemaT]):
    """
    A client for interacting with LLM APIs, including OpenAI-compatible
    and Hugging Face Transformers models, with structured output capabilities.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        accelerator,
        backend: Literal["deepinfra", "together", "openai", "huggingface"],
        quantization: str,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env_var: Optional[str] = None,
        # Hugging Face specific arguments
        hf_trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a new LLMClient.

        Args:
            model_name: Name of the LLM model to use.
            backend: The LLM backend to use ("openai" or "huggingface").
            api_base_url: Base URL for the OpenAI-compatible API (required for "openai" backend).
            api_key: API key for OpenAI authentication (optional if api_key_env_var is provided).
            api_key_env_var: Environment variable name containing the OpenAI API key.
            hf_trust_remote_code: Whether to trust remote code when loading Hugging Face models (use with caution).
            **kwargs: Additional keyword arguments for the chosen backend.
        """
        self.model_name = model_name
        self.backend = backend
        self.quantization = quantization
        self.client: Any = None  # This will hold the backend-specific client or model
        self.tokenizer: Any = tokenizer  # For Hugging Face models
        self.model: Any = model  # For Hugging Face models
        self.accelerator = accelerator

        if self.backend == "openai":
            if api_base_url is None:
                raise ValueError("api_base_url is required for 'openai' backend.")

            if api_key is None and api_key_env_var is not None:
                api_key = os.getenv(api_key_env_var)
                if not api_key:
                    raise ValueError(
                        f"[LLMClient] {api_key_env_var} is not set in the environment. "
                        f"Please set it as `export {api_key_env_var}=<your_api_key>`."
                    )
            elif api_key is None and api_key_env_var is None:
                raise ValueError(
                    "Either api_key or api_key_env_var must be provided for 'openai' backend."
                )

            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base_url,
            )
            logger.info(f"Initialized OpenAI LLMClient with model: {model_name}")

        elif self.backend == "huggingface":
            try:
                if not self.model:
                    logger.info(f"Loading Hugging Face model: {model_name}")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=hf_trust_remote_code,
                        device_map="auto",
                    )
                    logger.info(f"Hugging Face model {model_name} loaded successfully.")

                if not self.tokenizer:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                if self.accelerator:
                    self.config = self.accelerator.unwrap_model(self.model).config
                else:
                    self.config = self.model.config
            except Exception as e:
                logger.error(f"Error loading Hugging Face model {model_name}: {e}")
                raise
        elif self.backend in ["deepinfra", "together", "openrouter"]:
            pass
        else:
            raise ValueError(f"Unsupported backend: {backend}.")

    async def call(
        self,
        prompt: str,
        system_prompt: str,
        output_schema: Type[OutputSchemaT],
        do_sample: bool = False,
        temperature: float = 0.0,
        additional_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,  # Allows passing extra args to specific backend calls
    ) -> OutputSchemaT:
        """
        Make a call to the LLM with structured output.

        Args:
            prompt: The user prompt to send to the LLM.
            system_prompt: The system prompt to use for the LLM.
            output_schema: Pydantic model defining the expected output schema.
            temperature: Temperature setting for the LLM (default: 0.0).
            additional_messages: Additional messages to include in the conversation (default: None).
            **kwargs: Additional keyword arguments specific to the backend's call method.

        Returns:
            The LLM response parsed according to the output schema.
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        if additional_messages:
            messages.extend(additional_messages)

        if self.backend == "openai":
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=output_schema,
                    temperature=temperature,
                    **kwargs,
                )
                return response
            except Exception as e:
                logger.error(f"Error in OpenAI LLM call: {e}")
                raise
        elif self.backend in ["deepinfra", "together", "openrouter"]:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model_name.lower(),
                "messages": messages,
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": 
                        {
                            "name": "AlignmentCheckOutputSchema",
                            "strict": True,
                            "schema": output_schema.model_json_schema(),
                        }
                },
                "provider": {
                    "allow_fallbacks": True,
                    "order": ["Together", "DeepInfra", "DeepSeek"],
                    "quantizations": [self.quantization, "fp16", "bf16", "fp32"],
                    "require_parameters": True
                },
            }

            response = requests.post(url, headers=headers, json=payload)
            try:
                formatted_response = json.loads(response.json()["choices"][0]["message"]["content"])
                return output_schema.model_validate(formatted_response)
            except Exception as e:
                print("=============")
                print(e)
                print(response.json())
                print(type(response.json()))
                print("=============")
                print(e)
        elif self.backend == "huggingface":
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Hugging Face model not initialized correctly.")
            texts = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer(texts, return_tensors="pt").to(
                self.model.device
            )
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer, vocab_size=self.config.vocab_size
            )
            grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
            compiled_grammar = grammar_compiler.compile_json_schema(
                output_schema.model_json_schema()
            )
            xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

            generated_text = ""
            try:
                hf_call_kwargs = {
                    "max_new_tokens": 1024,
                    "num_return_sequences": 1,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "logits_processor": [xgr_logits_processor],
                }
                if temperature > 0:
                    hf_call_kwargs["do_sample"] = True
                hf_call_kwargs.update(kwargs)
                if self.accelerator:
                    generated_ids = self.accelerator.unwrap_model(self.model).generate(
                        **model_inputs, **hf_call_kwargs
                    )
                else:
                    generated_ids = self.model.generate(
                        **model_inputs, **hf_call_kwargs
                    )

                model_inputs.input_ids[0]
                generated_ids[0]
                generated_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                try:
                    json_output = json.loads(generated_text)
                except json.decoder.JSONDecodeError:
                    json_output = {
                        "observation": "",
                        "thought": "",
                        "conclusion": True
                    }
                return output_schema.model_validate(json_output)

            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Error parsing Hugging Face LLM output: {e}")
                logger.error(f"Generated Text (full): {generated_text}")
                raise
            except Exception as e:
                logger.error(f"Error in Hugging Face LLM call: {e}")
                raise
        else:
            raise RuntimeError("LLMClient not initialized with a valid backend.")
