import streamlit as st
from langchain.llms import OpenAI


def init_sidebar() -> None:
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
    st.session_state["openai_api_key"] = openai_api_key
