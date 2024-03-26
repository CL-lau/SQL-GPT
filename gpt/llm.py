import re
from typing import Optional

import streamlit as st
from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from embedding.embeddingHelper import SummaryEmbeddingHelper
from sql.SQLOperator import SQLHelper
from sql.SQL_type import get_db_operation_class, SQL_class
from sql.orm import orm


class LLM:
    def __init__(self, openai_api_key: Optional[str] = None, temperature: Optional[float] = 0.7):
        if openai_api_key is None:
            openai_api_key = st.session_state["openai_api_key"]
        if temperature is None:
            temperature = st.session_state["temperature"]

        self.llm = OpenAI(temperature=temperature, openai_api_key=openai_api_key)
        self.chatModel = ChatOpenAI(openai_api_key=openai_api_key)

    def generate_response(self, input_text: str, llm: Optional[OpenAI] = None, openai_api_key: Optional[str] = None, temperature: Optional[float] = None) -> str:
        if llm is None:
            if openai_api_key is None:
                openai_api_key = st.session_state["openai_api_key"]
            if temperature is None:
                temperature = st.session_state["temperature"]

        # st.info(self.llm(input_text))
        res = self.chatModel.invoke(input_text).content
        return res


class FileGPT:
    def __init__(self, fileId: Optional[str] = None, fileEmbeddingDB: Optional[str] = None):
        self.fileId = fileId
        self.fileEmbeddingDB = fileEmbeddingDB
        self.embeddingHelper = SummaryEmbeddingHelper()
        self.llm = LLM()

        self.file_prompt = "请回答一下问题{}，下面是对这个问题相关内容的提示{}"

        self.prompt_template = (
            "Only answer what is asked. Answer step-by-step.\n"
            "If the content has sections, please summarize them "
            "in order and present them in a bulleted format.\n"
            "Utilize line breaks for better readability.\n"
            "For example, sequentially summarize the "
            "introduction, methods, results, and so on.\n"
            "Please use Python's newline symbols appropriately to "
            "enhance the readability of the response, "
            "but don't use two newline symbols consecutive.\n\n"
            "{context}\n\n"
            "Question: {question}\n"
        )
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        self.refine_prompt_template = (
            "The original question is as follows: {question}\n"
            "We have provided an existing answer: {existing_answer}\n"
            "We have the opportunity to refine the existing answer"
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{context_str}\n"
            "------------\n"
            "Given the new context, refine the original answer to better "
            "answer the question. "
            "If the context isn't useful, return the original answer.\n"
            "Please use Python's newline symbols "
            "appropriately to enhance the readability of the response, "
            "but don't use two newline symbols consecutive.\n"
        )
        self.refine_prompt = PromptTemplate(
            template=self.refine_prompt_template,
            input_variables=['question', 'existing_answer', 'context_str']
        )

    def add_file(self, fileID: Optional[str] = None, docs: list[Document] = None) -> str:
        if fileID is None:
            fileID = self.embeddingHelper.from_documents(documents=docs, collectionId=None)
        else:
            self.embeddingHelper.from_documents(documents=docs, collectionId=fileID)
        return fileID

    def add_file_by_texts(self, fileID: Optional[str] = None, texts: list[str] = None) -> str:
        if fileID is None:
            fileID = self.embeddingHelper.from_texts(texts=texts, collectionId=None)
        else:
            self.embeddingHelper.from_texts(texts=texts, collectionId=fileID)
        return fileID

    def askQuestion(self, question: Optional[str], topK: Optional[int] = 4, fileID: Optional[str] = None):
        # TODO 查询embedding
        # db = Chroma.from_texts(documents, OpenAIEmbeddings())
        # docs = db.similarity_search(query=question, k=topK)
        docs = self.embeddingHelper.similarity_search(query=question, collectionId=fileID, topK=topK)

        question = self.prompt.format(context=docs, question=question)
        return self.llm.generate_response(input_text=question)


class SqlGPT:
    def __init__(self, sqlUrl: Optional[str], userName: Optional[str], password: Optional[str]):
        self.sqlUrl = sqlUrl
        self.userName = userName
        self.password = password

        self.llm = LLM()

        self.sqlHelper = SQLHelper()
        self.orm = orm()

        self.sql_prompt_template = PromptTemplate.from_template(
            "You now need to act as an SQL command intelligence generator, my current requirement is {question}. "
            "You only need to give the specific SQL command, you don't need to give any explanation."
        )

        self.optimize_prompt_template = PromptTemplate.from_template(
            "You now need to act as an SQL command optimizer and optimize the following sql statement {sql}, "
            "the structure of the associated table is {structure}, and the index of the associated table is {index}. "
            "Index optimization: Add appropriate indexes to your QL statements to improve query efficiency. "
            "SQL statement refactoring: Optimize the structure of SQL statements to reduce query time"
        )
        self.error_prompt_template = PromptTemplate.from_template(
            "The input sql statement is {sql}, the error is {error}, "
            "and the structure of the associated database is {structure}."
        )

    def generateSQL(self, question: Optional[str], dbID: Optional[str], need_operate: Optional[bool] = False, only_sql: Optional[bool] = True):
        question = self.sql_prompt_template.format(question=question)

        sql = self.llm.generate_response(input_text=question)

        sql = self.processSQL(sql)

        # only for update and select and insert
        if need_operate:
            operator_type = get_db_operation_class(sql)
            if operator_type == SQL_class.SELECT or operator_type == SQL_class.INSERT:
                result = self.sqlHelper.operate(sql=sql, dbID=dbID)
                return sql, result
        return sql

    def SQL_ERROR_CHECK(self, sql: Optional[str], error: Optional[str], need_operate: Optional[bool] = False, only_sql: Optional[bool] = True, dbID: Optional[str] = None):
        question = self.error_prompt_template.format(sql=sql, error=error, structure="")

        sql = self.llm.generate_response(input_text=question)

        if need_operate:
            result = self.sqlHelper.operate(sql)
            return sql, result
        return sql

    def optimizeSQL(self, sql: Optional[str], only_sql: Optional[bool] = True, dbID: Optional[str] = None):
        db_structure, db_index = self.SQL_operator.get_db_structure_and_index(sql)
        question = self.optimize_prompt_template.format(sql=sql, structure=db_structure, index=db_index)

        sql = self.llm.generate_response(input_text=question)
        return sql

    def processSQL(self, sql: Optional[str], dbID: Optional[str] = None):
        if str(sql).__contains__('\n'):
            sql = str(sql).replace('\n', ' ')
        self.orm.save(sql)
        res = ""

        # 匹配SQL语句
        pattern = r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b.*?(?=;|$)'
        sql_list = re.findall(pattern, sql, re.IGNORECASE | re.DOTALL)
        # 输出匹配结果
        for item in sql_list:
            res = res + item
            res = res + " "
        return sql

    def operate_sql(self, sql: Optional[str], dbID: Optional[str] = None):
        tag, res = self.sqlHelper.operate(sql=sql, dbID=dbID)
        print(res.head())

