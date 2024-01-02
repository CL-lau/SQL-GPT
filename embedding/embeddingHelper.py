import concurrent.futures
import logging
import os.path
import random
import string
import threading
import time
from typing import Optional

import numpy as np
import torch
import langchain
import chromadb
import torch.nn as nn
from langchain.text_splitter import TokenTextSplitter, MarkdownHeaderTextSplitter
from langchain.document_loaders import TextLoader, JSONLoader, UnstructuredMarkdownLoader, DirectoryLoader, \
    PythonLoader, OnlinePDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

from chromadb.utils import embedding_functions
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain.llms import OpenAI

from memory.RedisCaching import RedisCaching
import asyncio
import threading
import streamlit as st


def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@singleton
class EmbeddingHelper:
    def __init__(self, embeddingPath='./embedding', embeddingFile='test', need_redis=False, embedding_type='default'):
        super().__init__()
        self.embeddingPath = embeddingPath
        self.embeddingFile = embeddingFile
        self.client = chromadb.PersistentClient(path=embeddingPath)
        # self.client = chromadb.Client()

        self.collections = self.client.list_collections()

        # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        #     model_name="all-MiniLM-L6-v2")
        default_ef = embedding_functions.DefaultEmbeddingFunction()

        self.embedding_type = embedding_type

        self.embedding_model = default_ef
        self.collection = self.client.get_or_create_collection(name=embeddingFile,
                                                               embedding_function=default_ef)

        self.image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

        self.need_redis = need_redis

        self.checkClient()

        if self.need_redis:
            self.redisCaching = RedisCaching()

    def query(self, text, topK=5, where=None, where_document=None):
        # where={"metadata_field": "is_equal_to_this"},
        # where_document={"$contains": "search_string"}
        if self.need_redis:
            starttime = time.time()
            res = self.query_preprocessing(text, top_k=topK)
            if res is not None:
                if len(res) >= 1:
                    return res
        if where is None:
            if where_document is None:
                result = self.collection.query(
                    query_texts=[text],
                    n_results=topK
                )
            else:
                result = self.collection.query(
                    query_texts=[text],
                    n_results=topK,
                )
        else:
            if where_document is None:
                result = self.collection.query(
                    query_texts=[text],
                    n_results=topK,
                    where=where
                )
            else:
                result = self.collection.query(
                    query_texts=[text],
                    n_results=topK,
                    where=where
                )

        result = self.processResult(query=[text], result=result)
        """
        result : [[]] 第一层为查询的数量，现在为1，第二层为每一次查询的相似度结果
        """
        return result

    def batchQuery(self, texts, topK=5, where=None, where_document=None):
        # where={"metadata_field": "is_equal_to_this"},
        # where_document={"$contains": "search_string"}
        if self.need_redis:
            res = self.batchQuery_preprocessing(texts)
            if res is not None:
                return res
        if where is None:
            if where_document is None:
                result = self.collection.query(
                    query_texts=texts,
                    n_results=topK
                )
            else:
                result = self.collection.query(
                    query_texts=texts,
                    n_results=topK,
                )
        else:
            if where_document is None:
                result = self.collection.query(
                    query_texts=texts,
                    n_results=topK,
                    where=where
                )
            else:
                result = self.collection.query(
                    query_texts=texts,
                    n_results=topK,
                    where=where
                )

        result = self.processResult(query=texts, result=result)
        """
        result : [[]] 第一层为查询的数量，第二层为每一次查询的相似度结果
        """
        return result

    def queryByEmbedding(self, embedding, topK=5, where=None, where_document=None):
        if where is None:
            if where_document is None:
                result = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=topK
                )
            else:
                result = self.collection.query(
                    query_texts=[embedding],
                    n_results=topK,
                )
        else:
            if where_document is None:
                result = self.collection.query(
                    query_texts=[embedding],
                    n_results=topK,
                    where=where
                )
            else:
                result = self.collection.query(
                    query_texts=[embedding],
                    n_results=topK,
                    where=where
                )

        result = self.processResult(result=result)
        return result[0]

    def batchQueryByEmbedding(self, embeddings, topK=5, where=None, where_document=None):
        if where is None:
            if where_document is None:
                result = self.collection.query(
                    query_embeddings=embeddings,
                    n_results=topK
                )
            else:
                result = self.collection.query(
                    query_embeddings=embeddings,
                    n_results=topK,
                )
        else:
            if where_document is None:
                result = self.collection.query(
                    query_embeddings=embeddings,
                    n_results=topK,
                    where=where
                )
            else:
                result = self.collection.query(
                    query_embeddings=embeddings,
                    n_results=topK,
                    where=where
                )

        result = self.processResult(result=result)
        return result

    def saveFile(self, fileName, filePath, chunk_size=500, chunk_overlap=0, text_splitter=None, autoSpliter=False,
                 csv_delimiter=",",
                 csv_quotechar='"', csv_fieldnames=[], json_jq_schema='.messages[].content', json_lines=True,
                 dir_glob="**/*.md",
                 dir_show_progress=True, dir_use_multithreading=False,
                 dir_text_loader_kwargs={'autodetect_encoding': True},
                 dir_silent_errors=True, dir_loader_cls=TextLoader):
        documents = self.split(
            fileName=fileName,
            filePath=filePath,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            text_splitter=text_splitter,
            autoSpliter=autoSpliter,
            csv_delimiter=csv_delimiter,
            csv_quotechar=csv_quotechar,
            csv_fieldnames=csv_fieldnames,
            json_jq_schema=json_jq_schema,
            json_lines=json_lines,
            dir_glob=dir_glob,
            dir_show_progress=dir_show_progress,
            dir_use_multithreading=dir_use_multithreading,
            dir_text_loader_kwargs=dir_text_loader_kwargs,
            dir_silent_errors=dir_silent_errors,
            dir_loader_cls=dir_loader_cls
        )
        print(documents)
        fileName_withPath = filePath + '/' + fileName
        index = self.collection.count()
        self.collection.add(
            documents=[document.page_content for document in documents],
            ids=['id_' + str(index + i) for i in range(documents.__len__())],
            metadatas=[{
                'fileName': fileName_withPath,
                'documentIndex': i
            } for i in range(documents.__len__())]
        )

    def split(self, fileName, filePath, chunk_size=500, chunk_overlap=0, text_splitter=None, autoSpliter=False,
              csv_delimiter=",",
              csv_quotechar='"', csv_fieldnames=[], json_jq_schema='.messages[].content', json_lines=True,
              dir_glob="**/*.md",
              dir_show_progress=True, dir_use_multithreading=False,
              dir_text_loader_kwargs={'autodetect_encoding': True},
              dir_silent_errors=True, dir_loader_cls=TextLoader):
        text_splitter = self.getSpliter(
            textSplit=text_splitter,
            fileName=fileName,
            autoSpliter=autoSpliter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        if text_splitter is None:
            text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        fileName_withPath = filePath + '/' + fileName
        if str(fileName).startswith('http') and not str(fileName).endswith('pdf'):
            # TODO 进行文件的下载以及
            pass
        if str(fileName_withPath).endswith("pdf"):
            if str(fileName).startswith('http'):
                pdf = OnlinePDFLoader(fileName).load()
            else:
                pdf = PyPDFLoader(fileName_withPath).load()
            documents = text_splitter.split_documents(pdf)
        elif str(fileName_withPath).endswith("txt"):
            txt = TextLoader(fileName_withPath).load()
            documents = text_splitter.split_documents(txt)
        elif str(fileName_withPath).endswith("md"):
            md = UnstructuredMarkdownLoader(fileName_withPath).load()
            documents = text_splitter.split_documents(md)
        elif str(fileName_withPath).endswith("csv"):
            csv = CSVLoader(
                file_path=fileName_withPath,
                csv_args={
                    'delimiter': csv_delimiter,
                    'quotechar': csv_quotechar,
                    'fieldnames': csv_fieldnames
                }
            ).load()
            documents = text_splitter.split_documents(csv)
        elif str(fileName_withPath).endswith("json"):
            jsonStr = JSONLoader(
                fileName_withPath,
                jq_schema=json_jq_schema,
                json_lines=json_lines
            ).load()
            documents = text_splitter.split_documents(jsonStr)
        elif str(fileName_withPath).endswith("html"):
            pdf = UnstructuredHTMLLoader(fileName_withPath).load()
            documents = text_splitter.split_documents(pdf)
        elif str(fileName_withPath).endswith("/"):
            if dir_glob.endswith('.py'):
                dir_loader_cls = PythonLoader
            dirs = DirectoryLoader(
                fileName_withPath,
                glob=dir_glob,
                loader_cls=dir_loader_cls,
                loader_kwargs=dir_text_loader_kwargs,
                show_progress=dir_show_progress,
                use_multithreading=dir_use_multithreading,
                silent_errors=dir_silent_errors
            ).load()
            doc_sources = ''
            for doc in dirs:
                doc_sources = doc_sources + doc.metadata['source'] + ' ' + '\n'
            logging.info('load from ' + doc_sources)
            documents = text_splitter.split_documents(dirs)
        elif os.path.split(fileName_withPath)[0].lower() in self.image_extensions:
            documents = None
            pass
        else:
            documents = None
            pass

        return documents

    def checkClient(self):
        # assert not self.client.heartbeat(), logging.error("the embedding factory is not available.")
        pass

    def getSpliter(self, textSplit, fileName, autoSpliter=False, chunk_size=500, chunk_overlap=0):
        self.checkClient()
        if textSplit is not None:
            return textSplit

        text_splitter = None
        if autoSpliter:
            # 根据文件的类型来实现spliter
            if str(fileName).endswith("txt"):
                text_splitter = CharacterTextSplitter(
                    separator="\n\n",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
            elif str(fileName).endswith("md"):
                headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
                text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            elif str(fileName).endswith("py"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
            elif str(fileName).endswith("go"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.GO)
            elif str(fileName).endswith("java"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.JAVA)
            elif str(fileName).endswith("php"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.PHP)
            elif str(fileName).endswith("js"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.JS)
            elif str(fileName).endswith("proto"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.PROTO)
            elif str(fileName).endswith("rst"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.RST)
            elif str(fileName).endswith("ruby"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.RUBY)
            elif str(fileName).endswith("scala"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.SCALA)
            elif str(fileName).endswith("swift"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.SWIFT)
            elif str(fileName).endswith("latex"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.LATEX)
            elif str(fileName).endswith("html"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.HTML)
            elif str(fileName).endswith("sol"):
                text_splitter = RecursiveCharacterTextSplitter.get_separators_for_language(Language.SOL)
        return text_splitter

    def processResult(self, query: list = None, query_vec: list = None, result=None):
        if self.need_redis and query is not None:
            self.insert_cahing(
                query=query,
                query_vec=query_vec,
                result=result
            )

        return result['documents']
        # TODO 做一些打点标记

    def query_preprocessing(self, text: str, top_k=3):
        if self.need_redis:
            if isinstance(text, str):
                vec = self.embedding_model([text])[0]
                res = []
                ans = self.redisCaching.query_cache(query=text, query_vec=np.array(vec), k=top_k)
                if ans is not None:
                    ans = str(ans).split('___')
                    res.append(ans)
                return res
        return None

    def batchQuery_preprocessing(self, texts, top_k=3):
        if self.need_redis:
            if isinstance(texts, list):
                res = []
                for text in texts:
                    vec = self.embedding_model([text])[0]
                    ans = self.redisCaching.query_cache(query=text, query_vec=np.array(vec), k=top_k)
                    if ans is not None:
                        ans = str(ans).split('___')
                        res.append(ans)
                if len(res) == len(texts):
                    return res
        return None

    def query_embedding_preprocessing(self, text, thresholds=.9):
        if self.need_redis:
            if isinstance(text, str):
                text = text

        return text

    def batchQuery_embedding_preprocessing(self, text):
        if self.need_redis:
            if isinstance(text, list):
                text = text

        return text

    def query_postprocessing(self, text: str):
        if self.need_redis:
            if isinstance(text, str):
                text = text
                # TODO 加入redis
        return text

    def batchQuery_postprocessing(self, text):
        if self.need_redis:
            if isinstance(text, list):
                text = text

        return text

    def query_embedding_postprocessing(self, text, thresholds=.9):
        if self.need_redis:
            if isinstance(text, str):
                text = text

        return text

    def batchQuery_embedding_postprocessing(self, text):
        if self.need_redis:
            if isinstance(text, list):
                text = text

        return text

    def transEmbedding(self, text, model_type, refreshModel=False):
        if refreshModel:
            self.embedding_model = self.getEmbeddingModel(model_type=model_type)
        return self.embedding_model([text])

    def getEmbeddingModel(self, model_type, OPENAI_API_KEY=None, OPENAI_API_BASE_PATH=None, Cohere_API_KEY=None,
                          Sentence_Transformersmodel_name=None, Cohere_model_name=None, Instructor_model_name=None,
                          Google_api_key=None, Google_model_name=None, HuggingFace_api_key=None,
                          HuggingFace_model_name=None):
        """
        model_type:
        [ default,
        all-MiniLM-L6-v2,
        Sentence Transformers,
        OpenAI,
        Cohere,
        Instructor models,
        Google PaLM API models,
        HuggingFace ]
        """
        logging.info(self.embedding_type)
        model = embedding_functions.DefaultEmbeddingFunction()
        if model_type == "all-MiniLM-L6-v2":
            model = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "Sentence Transformers":
            model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=Sentence_Transformersmodel_name)
        elif model_type == "OpenAI":
            model = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                api_base=OPENAI_API_BASE_PATH,
                model_name="text-embedding-ada-002"
            )
        elif model_type == "Cohere":
            model = embedding_functions.CohereEmbeddingFunction(api_key=Cohere_API_KEY, model_name=Cohere_model_name)
        elif model_type == "Instructor models":
            if torch.cuda.is_available():
                model = embedding_functions.InstructorEmbeddingFunction(
                    model_name=Instructor_model_name, device="cuda")
            else:
                model = embedding_functions.InstructorEmbeddingFunction()
        elif model_type == "Google PaLM API models":
            model = embedding_functions.GooglePalmEmbeddingFunction(api_key=Google_api_key)
        elif model_type == "HuggingFace":
            model = embedding_functions.HuggingFaceEmbeddingFunction(
                api_key=HuggingFace_api_key,
                model_name=HuggingFace_model_name
            )
        return model

    def insert_cahing(self, query: list = None, query_vec: list = None, result=None):
        # default_ef = embedding_functions.DefaultEmbeddingFunction()
        query_vec = None
        for i in range(len(query)):
            mem = []
            for j in range(result['documents'][i].__len__()):
                mem.append((result['documents'][i][j], np.random.rand(384).tolist()))
                # mem.append((result['documents'][i][j], self.embedding_model(result['documents'][i][j])))

            if query_vec is None:
                vec = self.embedding_model(query[i])[0]
            if query_vec is not None:
                vec = query_vec[i]
            self.redisCaching.update_cache(query[i], vec, mem)

    def cahing_finished_callback(self, future):
        self.my_property = future.result()
        print("Long running task finished.")


@singleton
class LangChainEmbeddingHelper:
    def __init__(self):
        # Map{dbId, db}
        self.dbMap = {}

    def from_documents(self, documents: list[Document], dbId: str, embedding=OpenAIEmbeddings(),
                       persist_directory: str = None):
        if persist_directory is None:
            persist_directory = "./embedding" + dbId
        db = Chroma.from_documents(documents, embedding=embedding, persist_directory=persist_directory)
        self.dbMap[dbId] = db

    def from_texts(self, texts: list[str], dbId: str, embedding=OpenAIEmbeddings(), persist_directory: str = None):
        if persist_directory is None:
            persist_directory = "./embedding" + dbId
        db = Chroma.from_texts(texts, embedding=embedding, persist_directory=persist_directory)
        self.dbMap[dbId] = db

    def similarity_search(self, query: str, fileID: str, topK: int = 4) -> list[str]:
        db = self.dbMap[fileID]
        docs = db.similarity_search(query, k=topK)
        texts = []
        for doc in docs:
            texts.append(doc.page_content)
        return texts


@singleton
class SummaryEmbeddingHelper:

    openaiAccessRate = 0

    def __init__(self, embedding_function: Optional[str] = None):
        import chromadb
        import streamlit as st

        self.client = chromadb.PersistentClient(path="./embedding")

        self.generated_ids = set()

        # Map{collectionId, collection}
        self.collectionMap = {}

        # Map{collectionID, idIndex}
        self.idIndexMap = {}

        self.api_key = st.session_state["openai_api_key"]
        self.chatModel = ChatOpenAI(openai_api_key=self.api_key)

        if embedding_function is None:
            self.embedding_model = embedding_functions.DefaultEmbeddingFunction()
        elif embedding_function == "openai":
            self.embedding_model = embedding_functions.OpenAIEmbeddingFunction(api_key=self.api_key)

        self.summaryPromptChinese = "请你对下面的段落进行总结,使用中文进行总结,具体的段落内容如下: {}"
        self.summaryPromptEnglish = "Please summarize the following paragraph in English, the specific content of the " \
                                    "paragraph is as follows: {}"

    def from_documents(self, documents: list[Document], collectionId: Optional[str], embedding: Optional = OpenAIEmbeddings(),
                       persist_directory: Optional[str] = None) -> str:
        if collectionId is None:
            collectionId = self.generated_id()
        collection = self.client.get_or_create_collection(collectionId)
        idIndex = 0

        texts = []
        ids = []
        embeddings = []
        for document in documents:
            text = document.page_content
            texts.append(text)
            embeddings.append(self.summaryEmbedding(text))
            ids.append("id" + str(idIndex))
            idIndex += 1

        collection.add(
            documents=texts,
            embeddings=embeddings,
            # metadatas=[{"source": "notion"}, {"source": "google-docs"}],  # filter on these!
            ids=ids,
        )

        self.collectionMap[collectionId] = collection
        self.idIndexMap[collectionId] = idIndex
        return collectionId

    def from_texts(self, texts: list[str], collectionId: Optional[str] = None, embedding: Optional = OpenAIEmbeddings(),
                   persist_directory: Optional[str] = None) -> str:
        if collectionId is None:
            collectionId = self.generated_id()
        collection = self.client.get_or_create_collection(collectionId)
        idIndex = 0

        ids = []
        embeddings = []
        for text in texts:
            embeddings.append(self.summaryEmbedding(text))
            ids.append("id" + str(idIndex))
            idIndex += 1

        collection.add(
            documents=texts,
            embeddings=embeddings,
            # metadatas=[{"source": "notion"}, {"source": "google-docs"}],  # filter on these!
            ids=ids,  # unique for each doc
        )

        self.collectionMap[collectionId] = collection
        self.idIndexMap[collectionId] = idIndex
        return collectionId

    def similarity_search(self, query: str, collectionId: str, topK: int = 4) -> list[str]:
        if collectionId not in self.collectionMap.keys():
            return []

        collection = self.collectionMap[collectionId]

        texts = collection.query(
            query_texts=[query],
            n_results=topK,
        )
        return texts["documents"][0]

    def summaryEmbedding(self, content: str):
        limit_access = st.session_state["openai_limit_access"]
        if not limit_access:
            res = self.chatModel.invoke(self.summaryPromptEnglish.format(content))
            embedding = self.embedding_model([res.content])
        else:
            self.openaiAccessRate += 2
            if self.openaiAccessRate >= 3:
                time.sleep(20)
                res = self.chatModel.invoke(self.summaryPromptEnglish.format(content))
                embedding = self.embedding_model([res.content])
                self.openaiAccessRate %= 3
            else:
                res = self.chatModel.invoke(self.summaryPromptEnglish.format(content))
                embedding = self.embedding_model([res.content])
        return embedding[0]

    def generated_id(self) -> str:
        characters = string.ascii_letters + string.digits

        while True:
            idt = ''.join(random.choice(characters) for _ in range(13))
            if idt not in self.generated_ids:
                self.generated_ids.add(idt)
                break
        return idt

