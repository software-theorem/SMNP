from threading import Thread
from typing import Iterator
from time import sleep
import openai
import torch
from transformers import pipeline, AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import requests
import time
import random
import logging
import requests
from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

class GPT:
    def __init__(self, args):
        self.openai_key = args.openai_key
        self.model_name = args.model_name_or_path
        self.logger = args.logger
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens  # 修正：添加 max_tokens 变量
        self.n_reasoning_paths = args.n_reasoning_paths  # 修正：添加 n_reasoning_paths 变量

    def ask(self, input, history=[], system_prompt=DEFAULT_SYSTEM_PROMPT):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }
        # print(self.temperature)
        # openai.api_key = self.openai_key
        message = [{"role": "system", "content": system_prompt}]
        for his in history:
            q, a = his
            message.append({"role": "user", "content": q})
            message.append({"role": "assistant", "content": a})

        message.append({"role": "user","content": input})

        data = {
            "model": self.model_name,
            "messages": message,
            "max_tokens": self.max_new_tokens,  # 修正：使用 self.max_tokens
            "seed": 1024,
            "n": self.n_reasoning_paths  # 修正：使用 self.n_reasoning_paths
        }

        self.logger.info("message:")
        self.logger.info(message)

        while True:
            response = requests.post("", headers=headers, json=data)

            if response.status_code == 200:
                responses = response.json()
                result = responses["choices"][0]["message"]["content"]  # 获取第一条回复

                self.logger.info("result:")
                self.logger.info(result)

                time.sleep(1)  # 避免连续请求过快
                return result.strip()
            else:
                self.logger.error(f"Error: {response.status_code} - {response.text}")
                time.sleep(random.uniform(1, 3))  # 等待后重试

class Deepseek:
    def __init__(self, args):
        self.Deepseek_api_key = args.deepseek_key
        self.base_url = "https://api.deepseek.com"
        self.model_name = args.model_name_or_path
        self.logger = args.logger
        self.temperature = args.temperature
        self.top_p = args.top_p

    def ask(self, input, history=[], system_prompt=DEFAULT_SYSTEM_PROMPT):
        client = openai.OpenAI(api_key=self.Deepseek_api_key, base_url=self.base_url)
        messages = [{"role": "system", "content": system_prompt}]

        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": input})

        self.logger.info('message:')
        self.logger.info(messages)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=False
        )

        result = response.choices[0].message.content
        self.logger.info('result:')
        self.logger.info(result)
        sleep(1)
        return result.strip()

class deepseekcode:
    def __init__(self, args):
        self.model_name = args.model_name_or_path
        self.logger = args.logger
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens
        self.n_reasoning_paths = args.n_reasoning_paths
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16  # 或换成 float16
        ).cuda()
        self.model.eval()

    def ask(self, input, history=[], system_prompt=DEFAULT_SYSTEM_PROMPT):
        messages = [{"role": "system", "content": system_prompt}]
        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": input})

        self.logger.info("message:")
        self.logger.info(messages)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()


        responses = []

        for _ in range(self.n_reasoning_paths):
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id

                )
            response_text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            responses.append(response_text)

            self.logger.info("response:")
            self.logger.info(response_text)

            time.sleep(1)  # 控制请求频率（可选）

        return responses if self.n_reasoning_paths > 1 else responses[0]

class qwen:
    def __init__(self, args):
        self.qwen_api_key = args.qwen_key
        self.base_url = "
        self.model_name = args.model_name_or_path
        self.logger = args.logger
        self.temperature = args.temperature
        self.top_p = args.top_p

    def ask(self, input, history=[], system_prompt=DEFAULT_SYSTEM_PROMPT):
        client = openai.OpenAI(api_key=self.qwen_api_key, base_url=self.base_url)
        messages = [{"role": "system", "content": system_prompt}]

        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": input})

        self.logger.info('message:')
        self.logger.info(messages)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=False
        )

        result = response.choices[0].message.content
        self.logger.info('result:')
        self.logger.info(result)
        sleep(1)
        return result.strip()

class qwencoder:
    def __init__(self, args):
        self.qwencoder_key = args.qwencoder_key
        self.base_url = ""
        self.model_name = args.model_name_or_path
        self.logger = args.logger
        self.temperature = args.temperature
        self.top_p = args.top_p

    def ask(self, input, history=[], system_prompt=DEFAULT_SYSTEM_PROMPT):
        client = openai.OpenAI(api_key=self.qwencoder_key, base_url=self.base_url)
        messages = [{"role": "system", "content": system_prompt}]

        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": input})

        self.logger.info('message:')
        self.logger.info(messages)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=False
        )

        result = response.choices[0].message.content
        self.logger.info('result:')
        self.logger.info(result)
        sleep(1)
        return result.strip()