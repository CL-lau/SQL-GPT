import os
from abc import ABC
from typing import Any

import simplejson
import streamlit as st
from langchain.chains.base import Chain
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import PromptTemplate

from sql.conversationORM import ConversationManager, Conversation

"""
Message的数据库存储以及st-session存储
"""


class ConversationMemory(BaseChatMemory, ABC):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.conversations = []

    def addConversation(self, question: str, answer: str):
        conversation = {"question": question, "answer": answer}
        self.conversations.append(conversation)

    def addQuestion(self, question: str):
        conversation = {"question": question}
        self.conversations.append(conversation)

    def addAnswer(self, answer: str):
        self.conversations[-1]["answer"] = answer

    def covert(self) -> str:
        return simplejson.dumps(self.conversations)


class RoleManage:
    def __init__(self):
        self.roleTypeList = ["system", "user", "assistant"]

    def roleCheck(self, role_type: str) -> bool:
        return role_type in self.roleTypeList

    def transRole(self, role_type: str) -> str:
        return "user"


class MessageManage:
    def __init__(self, conversation_ids: list[str]):
        self.conversation_ids = conversation_ids

        self.conversation_map = {}
        for conversation_id in self.conversation_ids:
            self.conversation_map[conversation_id] = ConversationManager.query_by_idWithSort(conversation_id=conversation_id)

        self.memory_map = {}
        for conversation_id, conversations in self.conversation_map.items():
            memory = ConversationMemory()
            for conversation in conversations:
                memory.addQuestion(conversation.content)
                memory.addAnswer(conversation.content)
            self.memory_map[conversation_id] = memory

        self.llmChain_map = {}
        for conversation_id, conversations in self.conversation_map.items():
            template = """You are a chatbot having a conversation with a human.

            {chat_history}
            Human: {human_input}
            Chatbot:"""

            prompt = PromptTemplate(
                input_variables=["chat_history", "human_input"], template=template
            )
            memory = ConversationBufferMemory(memory_key="chat_history")
            llm = OpenAI(
                temperature=st.session_state["temperature"],
                openai_api_key=st.session_state["openai_api_key"])
            llm_chain = LLMChain(
                llm=llm,
                prompt=prompt,
                verbose=True,
                memory=memory,
            )
            self.llmChain_map[conversation_id] = llm_chain

    def add_conversation_id(self, conversation_id: str):
        conversations = ConversationManager.query_by_idWithSort(conversation_id=conversation_id)
        self.conversation_map[conversation_id] = conversations

    def add_conversation(self, conversation_id: str, role_name: str, content: str):
        conversation_id_res = ConversationManager.add(conversation_id=conversation_id, role_name=role_name, content=content)
        conversation_id_tmp = ConversationManager.query_latest_id(conversation_id=conversation_id)
        if conversation_id_res == conversation_id_tmp:
            self.conversation_map[conversation_id] = ConversationManager.query_by_idWithSort(conversation_id=conversation_id)


