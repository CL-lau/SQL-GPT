import subprocess
import sys
from typing import Optional

import streamlit as st

from components import upload_and_process_document
from gpt.llm import FileGPT, SqlGPT
from utils.st_content import init_upload_dir, init_sidebar, init_proxy, init_openai_limit


def file_chat(fileId: Optional[str], fileGPT: Optional[FileGPT] = None):
    # client = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    if fileGPT is None:
        fileGPT = FileGPT()

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        # TODO 加载历史会话
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = fileGPT.askQuestion(question=question, fileID=fileId)

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def sql_chat(dbId: Optional[str], sqlGPT: Optional[SqlGPT] = None):
    if sqlGPT is None:
        sqlUrl = st.session_state["openai_model"]
        sqlGPT = SqlGPT()

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        # TODO 加载历史会话
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = sqlGPT.generateSQL(question=question)

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def main():
    st.title('🦜🔗 Quickstart App')

    init_upload_dir()
    init_sidebar()
    init_proxy()
    init_openai_limit()
    tag, res = upload_and_process_document()

    if tag == 1:
        [docs] = res
        fileGPT = FileGPT()
        fileID = fileGPT.add_file(docs=docs)
        file_chat(fileID)
    elif tag == 2:
        [sql_url, sql_username, sql_password] = res
        # fileGPT = FileGPT()
    elif tag == 3:
        [code_url] = res
        # fileGPT = FileGPT()
    elif tag == 4:
        pass
        # [text] = res
        # fileGPT = FileGPT()
    else:
        file_chat()


if __name__ == '__main__':
    main()

