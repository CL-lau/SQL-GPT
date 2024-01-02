import os
from abc import ABC, abstractmethod

import openai
import streamlit as st


class ApiKey(ABC):
    """Check the Api key is valid or not"""
    query = 'This is a test.'

    @classmethod
    @abstractmethod
    def is_valid(cls):
        pass


class OpenAiAPI(ApiKey):
    @classmethod
    def is_valid(cls) -> str:
        if not st.session_state['openai_api_key']:
            st.error('⚠️ :red[You have not pass OpenAI API key.] Use default model')
            return

        openai.api_key = os.getenv('OPENAI_API_KEY')
        try:
            response = openai.Completion.create(
                engine='davinci',
                prompt=cls.query,
                max_tokens=5
            )
            return response
        except Exception as e:
            st.error(
                '🚨 :red[Your OpenAI API key has a problem.] '
                '[Check your usage](https://platform.openai.com/account/usage)'
            )
            print(f'Test error\n{e}')


class SerpAPI(ApiKey):
    @classmethod
    def is_valid(cls) -> str:
        if not st.session_state['serpapi_api_key']:
            st.warning('⚠️ You have not pass SerpAPI key. (You cannot ask current events.)')
            return
        from langchain import SerpAPIWrapper

        os.environ['SERPAPI_API_KEY'] = os.getenv('SERPAPI_API_KEY')
        try:
            search = SerpAPIWrapper()
            response = search.run(cls.query)
            return response
        except Exception as e:
            st.error(
                '🚨 :red[Your SerpAPI key has a problem.] '
                '[Check your usage](https://serpapi.com/dashboard)'
            )
            print(f'Test error\n{e}')
