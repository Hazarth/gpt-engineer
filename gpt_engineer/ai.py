from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Dict, List

import tiktoken

from gpt4all import GPT4All

FORMAT_ALPACA = {
    "system": "### Instruction:",
    "prompt": "### Input:",
    "response": "### Response:",
}

FORMAT = FORMAT_ALPACA

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    step_name: str
    in_step_prompt_tokens: int
    in_step_completion_tokens: int
    in_step_total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int


class AI:
    model: GPT4All = None

    def __init__(self, model, temperature=0.1):
        self.temperature = temperature

        # initialize token usage log
        self.cumulative_prompt_tokens = 0
        self.cumulative_completion_tokens = 0
        self.cumulative_total_tokens = 0
        self.token_usage_log = []

        try:
            if not AI.model:
                AI.model = GPT4All(model)
        except Exception:
            print("Unexpected error")

        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.debug(
                f"Tiktoken encoder for model {model} not found. Using "
                "cl100k_base encoder instead. The results may therefore be "
                "inaccurate and should only be used as estimate."
            )
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def start(self, system, user, step_name):
        messages = [
            {"role": "system", "content": f"{FORMAT['system']}: {system}"},
            {"role": "user", "content": user},
        ]

        return self.next(messages, step_name=step_name)

    def fsystem(self, msg):
        return {"role": "system", "content": msg}

    def fuser(self, msg):
        return {"role": "user", "content": msg}

    def fassistant(self, msg):
        return {"role": "assistant", "content": msg}

    def next(self, messages: List[Dict[str, str]], prompt=None, *, step_name=None):
        if FORMAT["system"] not in messages[0]["content"]:
            messages[0]["content"] = FORMAT["system"] + " " + messages[0]["content"]
        if prompt:
            messages = messages + [
                {"role": "user", "content": f"{FORMAT['prompt']} \n{prompt}"}
            ]
        logger.debug(f"Creating a new chat completion: {messages}")

        response = AI.model.chat_completion(
            messages=messages,
            verbose=True,
            streaming=True,
            default_prompt_header=False,
            n_ctx=32768,
            n_predict=4096,
            temp=self.temperature,
        )

        logger.debug(f"Chat completion finished: {messages}")
        chat = response["choices"][0]["message"]["content"]
        messages = messages + [response["choices"][0]["message"]]

        self.update_token_usage_log(
            messages=messages, answer="".join(chat), step_name=step_name
        )

        return messages

    def update_token_usage_log(self, messages, answer, step_name):
        prompt_tokens = self.num_tokens_from_messages(messages)
        completion_tokens = self.num_tokens(answer)
        total_tokens = prompt_tokens + completion_tokens

        self.cumulative_prompt_tokens += prompt_tokens
        self.cumulative_completion_tokens += completion_tokens
        self.cumulative_total_tokens += total_tokens

        self.token_usage_log.append(
            TokenUsage(
                step_name=step_name,
                in_step_prompt_tokens=prompt_tokens,
                in_step_completion_tokens=completion_tokens,
                in_step_total_tokens=total_tokens,
                total_prompt_tokens=self.cumulative_prompt_tokens,
                total_completion_tokens=self.cumulative_completion_tokens,
                total_tokens=self.cumulative_total_tokens,
            )
        )

    def format_token_usage_log(self):
        result = "step_name,"
        result += "prompt_tokens_in_step,completion_tokens_in_step,total_tokens_in_step"
        result += ",total_prompt_tokens,total_completion_tokens,total_tokens\n"
        for log in self.token_usage_log:
            result += log.step_name + ","
            result += str(log.in_step_prompt_tokens) + ","
            result += str(log.in_step_completion_tokens) + ","
            result += str(log.in_step_total_tokens) + ","
            result += str(log.total_prompt_tokens) + ","
            result += str(log.total_completion_tokens) + ","
            result += str(log.total_tokens) + "\n"
        return result

    def num_tokens(self, txt):
        return len(self.tokenizer.encode(txt))

    def num_tokens_from_messages(self, messages):
        """Returns the number of tokens used by a list of messages."""
        n_tokens = 0
        for message in messages:
            n_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                n_tokens += self.num_tokens(value)
                if key == "name":  # if there's a name, the role is omitted
                    n_tokens += -1  # role is always required and always 1 token
        n_tokens += 2  # every reply is primed with <im_start>assistant
        return n_tokens


def fallback_model(model: str) -> str:
    try:
        AI.model = GPT4All(model)
        return model
    except Exception:
        print(f"Model {model} not available. Reverting " "to ggml-replit-code-v1-3b.bin")
        return "ggml-replit-code-v1-3b.bin"
