import torch
import torch.nn as nn

from embedding.embeddingHelper import EmbeddingHelper
from gpt.chat import ChatGPT


class File_GPT(ChatGPT):
    def __init__(self):
        super().__init__(OPENAI_API_KEY="", OPENAI_API_BASE="")
        self.embeddingHelper = EmbeddingHelper()
        self.file_prompt = "请回答一下问题{}，下面是对这个问题相关内容的提示{}"

    def askFile(self, question, need_redis=False, top_k=3):
        assisants_str = ""
        assisants = self.embeddingHelper.query(text=question, topK=top_k)
        for assisant in assisants:
            assisants_str += assisant

        ans = self.chat(questions=self.file_prompt.format(question, assisants_str), system_assistant=self.SQL_prompt)
        return ans
