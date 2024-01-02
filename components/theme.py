import streamlit as st


def theme() -> None:
    st.set_page_config(page_title="Document GPT")
    st.image('./static/img/chatbot_v2.png', width=150)
