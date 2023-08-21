import torch
import torch.nn as nn

from embedding.embeddingHelper import EmbeddingHelper
from gpt.chat import ChatGPT


class File_GPT(ChatGPT):
    def __init__(self, embeddingPath='./embedding',
                 embeddingFile='test',
                 need_redis=False,
                 embedding_type='default'
                 ):
        super().__init__(OPENAI_API_KEY="", OPENAI_API_BASE="")
        self.embeddingHelper = EmbeddingHelper(
            embeddingPath=embeddingPath,
            embeddingFile=embeddingFile,
            need_redis=need_redis,
            embedding_type=embedding_type
        )
        self.file_prompt = "请回答一下问题{}，下面是对这个问题相关内容的提示{}"

    def askFile(self, question, top_k=3):
        assisants = self.embeddingHelper.query(text=question, topK=top_k)
        assisants_str = " ".join(assisants[0])

        ans = self.chat(questions=self.file_prompt.format(question, assisants_str))
        return ans

    def addFile(self, fileName, filePath):
        self.embeddingHelper.saveFile(fileName=fileName, filePath=filePath)
