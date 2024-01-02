import streamlit as st
import os
from langchain.llms import OpenAI


def init_upload_dir() -> None:
    if not os.path.exists("uploads"):
        os.makedirs("uploads")


def init_sidebar() -> None:
    # st.sidebar.markdown(
    #     "### 输入API-KEY"
    # )

    openai_api_key = st.sidebar.text_input(
        'Input API-KEY',
        type='password',
        # label_visibility='hidden'
    )
    if openai_api_key is not None:
        if openai_api_key.startswith("sk-"):
            st.session_state["openai_api_key"] = openai_api_key
    else:
        OPENAI_API_KEY = os.environ.get('openai_api_key')
        if OPENAI_API_KEY is not None:
            if OPENAI_API_KEY.startswith("sk-"):
                st.session_state["openai_api_key"] = OPENAI_API_KEY

        # st.sidebar.markdown(
    #     "### 选择temperature"
    # )

    values = st.sidebar.slider(
        'Select OpenAI temperature',
        0.0, 1.0, 0.7,
        step=0.1,
        format="%f",
        # label_visibility='hidden'
    )
    st.session_state["temperature"] = values


def init_proxy() -> None:
    st.sidebar.markdown(
        """
        ###
        """
    )
    proxy_address = st.sidebar.text_input(
        'Input proxy address',
        placeholder='127.0.0.1',
        # label_visibility='hidden'
    )

    proxy_port = st.sidebar.number_input(
        'Input proxy port',
        # placeholder='7890',
        step=1,
        min_value=0,
        max_value=65535,
        format="%d",
        # label_visibility='hidden'
    )
    with st.sidebar.status("设置代理...", expanded=True) as status:
        if is_valid_ip(proxy_address) and is_valid_port(proxy_port):
            if proxy_address.strip() != "" and proxy_port.strip() != "":
                os.environ['http_proxy'] = f'http://{proxy_address}:{proxy_port}'
                os.environ['https_proxy'] = f'http://{proxy_address}:{proxy_port}'
                status.update(label="代理设置完成", state="complete", expanded=False)
            else:
                status.update(label="未设置代理", state="error", expanded=False)
        else:
            status.update(label="未设置代理", state="error", expanded=False)


def init_openai_limit() -> None:
    st.session_state["openai_limit_access"] = False

    on = st.sidebar.toggle('Limit Access')
    if on:
        st.session_state["openai_limit_access"] = True


def is_valid_ip(ip):
    import re
    pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    if re.match(pattern, ip):
        return True
    else:
        return False


def is_valid_port(port):
    try:
        port = int(port)
        if 0 <= int(port) <= 65535:
            return True
        else:
            return False
    except:
        return False
