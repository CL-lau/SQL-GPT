import torch
import langchain
import chromadb
import torch.nn as nn
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


class EmbeddingHelper(nn.Module):
    def __init__(self, embeddingPath, embeddingFile):
        super().__init__()
        self.embeddingPath = embeddingPath
        self.embeddingFile = embeddingFile
        client = chromadb.PersistentClient(path=embeddingPath)

        collection = client.create_collection(name=embeddingFile)

    def split(self, fileName, filePath, chunk_size=500, chunk_overlap=0):
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        fileName = filePath + '/' + filePath
        if str(fileName).endswith("pdf"):
            pdf = PyPDFLoader(fileName).load()
            documents = text_splitter.split_documents(pdf)

