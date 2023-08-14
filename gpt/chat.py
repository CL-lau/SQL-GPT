import json
import logging
import os
import openai
import torch
import torch.nn as nn
import tiktoken
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(module)s - %(message)s', level=logging.INFO)


class ChatGPT(nn.Module):
    def __init__(self, OPENAI_API_KEY=None, OPENAI_API_BASE=None, MODEL_TYPE="gpt-3.5-turbo"):
        super().__init__()
        self.config_file = "config.json"
        self.app_keys = []
        if OPENAI_API_KEY is None or OPENAI_API_KEY == '':
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
            if OPENAI_API_KEY is not None and OPENAI_API_KEY != "":
                openai.api_key = OPENAI_API_KEY
            self.OPENAI_API_KEY = OPENAI_API_KEY
        if OPENAI_API_BASE is None or OPENAI_API_BASE == '':
            OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
            if OPENAI_API_BASE is not None and OPENAI_API_BASE != "":
                openai.api_base = OPENAI_API_BASE
                self.OPENAI_API_BASE = OPENAI_API_BASE
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.OPENAI_API_BASE = OPENAI_API_BASE
        self.MODEL_TYPE = MODEL_TYPE
        self.MAX_TOKEN = 8000
        self.contextual_conversations = []

        self.initConfig()
        if self.OPENAI_API_KEY is None and len(self.app_keys) > 0:
            self.find_valid_key()

    def initConfig(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as config_file:
                config = json.load(config_file)
            openai_config = config['openai']
            proxy_config = config['proxy']

            openai_app_key = openai_config['app_key']
            openai_url_base = openai_config['url_base']
            openai_app_keys = openai_config['app_keys']
            model_type = openai_config['model_type']

            proxy_address = proxy_config['address']
            proxy_port = proxy_config['port']
            if openai_app_key is not None and openai_app_key != "":
                openai.api_base = openai_app_key
                self.OPENAI_API_KEY = openai_app_key
            if openai_url_base is not None and openai_url_base != "":
                openai.api_base = openai_url_base
                self.OPENAI_API_BASE = openai_url_base
            if openai_app_keys is not None and len(openai_app_keys) > 0:
                for key in openai_app_keys:
                    self.app_keys.append(key)
            if model_type is not None and model_type != "":
                self.MODEL_TYPE = model_type
            if proxy_address is not None and proxy_address != "" and proxy_port is not None and proxy_port != "":
                os.environ['http_proxy'] = f'http://{proxy_address}:{proxy_port}'
                os.environ['https_proxy'] = f'http://{proxy_address}:{proxy_port}'

    def find_valid_key(self):
        for key in self.app_keys:
            openai.api_key = key
            try:
                openai.ChatCompletion.create(model=self.MODEL_TYPE,
                                             messages=[
                                                 {"role": "system",
                                                  "content": "You are a helpful assistant."}
                                             ])
                self.app_keys.remove(key)
                self.app_keys.insert(0, key)
                return key
            except openai.error.AuthenticationError:
                logging.info(f"Key {key} is not valid.")
            except openai.error.OpenAIError as e:
                logging.info(f"Error while testing key {key}: {e}")
        logging.error("No valid keys found.")
        return None

    def chat(self, questions, system_assistant=None, assistant=None, temperature=None, need_stream=False):
        try:
            messages = []
            if system_assistant is not None:
                messages.append({"role": "system", "content": system_assistant})
            if assistant is not None:
                messages.append({"role": "assistant", "content": assistant})
            if isinstance(questions, list):
                for index, question in enumerate(questions):
                    messages.append({"role": "user", "content": question})
            if isinstance(questions, str):
                messages.append({"role": "user", "content": questions})
            response = None
            if temperature is None:
                response = openai.ChatCompletion.create(
                    model=self.MODEL_TYPE,
                    messages=messages,
                )
            else:
                if isinstance(temperature, float):
                    response = openai.ChatCompletion.create(
                        model=self.MODEL_TYPE,
                        messages=messages,
                        temperature=temperature,
                    )
            result = self.processResponse(response)
            return result
        except openai.error.AuthenticationError:
            if len(self.app_keys) > 0:
                available_key = self.find_valid_key()
                openai.api_key = self.app_keys[0]
                self.OPENAI_API_KEY = self.app_keys[0]
                self.chat(questions, system_assistant, assistant, temperature, need_stream)
                if available_key is None:
                    return ""
            else:
                logging.error("No valid keys found.")
                return ""
        except openai.error.OpenAIError as e:
            if len(self.app_keys) > 0:
                available_key = self.find_valid_key()
                openai.api_key = self.app_keys[0]
                self.OPENAI_API_KEY = self.app_keys[0]
                self.chat(questions, system_assistant, assistant, temperature, need_stream)
                if available_key is None:
                    return ""
            else:
                logging.error("No valid keys found.")
                return ""

    def contextual_chat(self, questions, system_assistant=None, assistant=None, temperature=None, need_stream=False):
        try:
            if system_assistant is not None:
                self.contextual_conversations.append({"role": "system", "content": system_assistant})
            if assistant is not None:
                self.contextual_conversations.append({"role": "assistant", "content": assistant})
            if isinstance(questions, list):
                for index, question in enumerate(questions):
                    self.contextual_conversations.append({"role": "user", "content": question})
            if isinstance(questions, str):
                self.contextual_conversations.append({"role": "user", "content": questions})
            response = None
            if temperature is None:
                response = openai.ChatCompletion.create(
                    model=self.MODEL_TYPE,
                    messages=self.contextual_conversations,
                )
            else:
                if isinstance(temperature, float):
                    response = openai.ChatCompletion.create(
                        model=self.MODEL_TYPE,
                        messages=self.contextual_conversations,
                        temperature=temperature,
                    )
            result = self.processResponse(response)
            self.contextual_conversations.append({"role": "assistant", "content": result})
            if self.num_tokens_from_messages(self.contextual_conversations) > self.MAX_TOKEN:
                self.contextual_conversations = []
            return result
        except openai.error.AuthenticationError:
            if len(self.app_keys) > 0:
                available_key = self.find_valid_key()
                openai.api_key = self.app_keys[0]
                self.OPENAI_API_KEY = self.app_keys[0]
                self.chat([], None, None, None, False)
                if available_key is None:
                    logging.error("No valid keys found in app key list.")
                    return ""
            else:
                logging.error("No valid keys found.")
                return ""
        except openai.error.OpenAIError as e:
            if len(self.app_keys) > 0:
                available_key = self.find_valid_key()
                openai.api_key = self.app_keys[0]
                self.OPENAI_API_KEY = self.app_keys[0]
                self.chat([], None, None, None, False)
                if available_key is None:
                    logging.error("No valid keys found.")
                    return ""
            else:
                logging.error("No valid keys found.")
                return ""

    def processResponse(self, response):
        """
        :param response:
        :return: result
            id: 请求的ID
            object：返回的对象类型（例如，chat.completion）
            created: 请求的时间戳
            model：用于生成 response 的模型的全名
            usage：用于生成回复、对prompt计数、完成completion 的 总计使用的 token 数
            choices：completion 的列表（只有一个，除非将 n 设置为大于 1）
            message：模型生成的消息对象，有role和content
            finish_reason：模型停止生成文本的原因（如果达到 max_tokens 限制，则为 stop 或 length）
            index: completion在 choices 列表中的索引
        """
        result = response['choices'][0]['message']['content']
        completion_tokens = response['usage']['completion_tokens']
        prompt_tokens = response['usage']['prompt_tokens']
        total_tokens = response['usage']['total_tokens']
        finish_reason = response['choices'][0]['finish_reason']
        index = response['choices'][0]['finish_reason']
        if total_tokens > self.MAX_TOKEN:
            logging.error("the input token exceed the max token.")
        logging.info("the token cost" + str(
            {"completion_tokens": completion_tokens,
             "prompt_tokens": prompt_tokens,
             "total_tokens": total_tokens,
             "finish_reason": finish_reason}))
        return result

    def num_tokens_from_messages(self, messages):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.MODEL_TYPE)
        except KeyError:
            print("Warning: model not f"
                  "ound. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if self.MODEL_TYPE == "gpt-3.5-turbo":
            print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
        elif self.MODEL_TYPE == "gpt-4":
            print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0314")
        elif self.MODEL_TYPE == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif self.MODEL_TYPE == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {self.MODEL_TYPE}. See https://github.com
                /openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
