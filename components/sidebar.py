import asyncio
import os

import streamlit as st

# from docGPT import GPT4Free


def side_bar() -> None:
    with st.sidebar:
        with st.expander(':orange[How to use?]'):
            st.markdown(
                """
                1. Enter your API keys: (You can use the `gpt4free` free model **without API keys**)
                    * `OpenAI API Key`: Make sure you still have usage left
                    * `SERPAPI API Key`: Optional. If you want to ask questions about content not appearing in the PDF document, you need this key.
                2. **Upload a Document** file (choose one method):
                    * method1: Browse and upload your own document file from your local machine.
                    * method2: Enter the document URL link directly.
                    
                    (**support documents**: `.pdf`, `.docx`, `.csv`, `.txt`)
                3. Start asking questions!
                4. More details.(https://github.com/Lin-jun-xiang/docGPT-streamlit)
                5. If you have any questions, feel free to leave comments and engage in discussions.(https://github.com/Lin-jun-xiang/docGPT-streamlit/issues)
                """
            )

    with st.sidebar:
        if st.session_state.openai_api_key:
            OPENAI_API_KEY = st.session_state.openai_api_key
            st.sidebar.success('API key loaded form previous input')
        else:
            OPENAI_API_KEY = st.sidebar.text_input(
                label='#### Your OpenAI API Key 👇',
                placeholder="sk-...",
                type="password",
                key='OPENAI_API_KEY'
            )
            st.session_state.openai_api_key = OPENAI_API_KEY

        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    with st.sidebar:
        if st.session_state.serpapi_api_key:
            SERPAPI_API_KEY = st.session_state.serpapi_api_key
            st.sidebar.success('API key loaded form previous input')
        else:
            SERPAPI_API_KEY = st.sidebar.text_input(
                label='#### Your SERPAPI API Key 👇',
                placeholder="...",
                type="password",
                key='SERPAPI_API_KEY'
            )
            st.session_state.serpapi_api_key = SERPAPI_API_KEY

        os.environ['SERPAPI_API_KEY'] = SERPAPI_API_KEY

    with st.sidebar:
        gpt4free = GPT4Free()
        st.session_state.g4f_provider = st.selectbox(
            (
                "#### Select a provider if you want to use free model. "
                "([details](https://github.com/xtekky/gpt4free#models))"
            ),
            (['BestProvider'] + list(gpt4free.providers_table.keys()))
        )

        st.session_state.button_clicked = st.button(
            'Show Available Providers',
            help='Click to test which providers are currently available.',
            type='primary'
        )
        if st.session_state.button_clicked:
            available_providers = asyncio.run(gpt4free.show_available_providers())
            st.session_state.query.append('What are the available providers right now?')
            st.session_state.response.append(
                'The current available providers are:\n'
                f'{available_providers}'
            )