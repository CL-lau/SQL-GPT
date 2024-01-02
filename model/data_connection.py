import os
from typing import Iterator, Union, List, Any

import requests
import streamlit as st
from langchain.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyMuPDFLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
    HTMLHeaderTextSplitter
)
from langchain_core.documents import Document


class DocumentLoader:
    @staticmethod
    def get_files(path: str, filetype: str = '.pdf') -> Iterator[str]:
        try:
            yield from [
                file_name for file_name in os.listdir(f'{path}')
                if file_name.endswith(filetype)
            ]
        except FileNotFoundError as e:
            print(f'\033[31m{e}')

    @staticmethod
    def load_documents(
        file: str,
        filetype: str = '.pdf'
    ) -> list[Document] | list[Any]:
    #) -> Union[CSVLoader, Docx2txtLoader, PyMuPDFLoader, TextLoader]:
        """Loading PDF, Docx, CSV"""
        try:
            if filetype == '.pdf':
                loader = PyMuPDFLoader(file)
            elif filetype == '.docx':
                loader = Docx2txtLoader(file)
            elif filetype == '.csv':
                loader = CSVLoader(file, encoding='utf-8')
            elif filetype == '.txt':
                loader = TextLoader(file, encoding='utf-8')

            return loader.load()

        except Exception as e:
            print(f'\033[31m{e}')
            return []

    @staticmethod
    def split_documents(
        document: list[Document],
        chunk_size: int=300,
        chunk_overlap: int=0
    ) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return splitter.split_documents(document)

    @staticmethod
    def split_documents_2texts(
            documents: list[Document],
            chunk_size: int = 300,
            chunk_overlap: int = 0
    ) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        content = ""
        for document in documents:
            content += document.page_content
        return splitter.split_text(content)

    @staticmethod
    def crawl_file(url: str) -> str:
        try:
            response = requests.get(url)
            filetype = os.path.splitext(url)[1]
            if response.status_code == 200 and (
                any(ext in filetype for ext in ['.pdf', '.docx', '.csv', '.txt'])
            ):
                if response.content.strip():
                    upload_file_name = url.split("/")[-1]
                    file_path = os.path.join("uploads", upload_file_name)
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    st.success(f"已保存文件: {upload_file_name}")
                return response.content, filetype
            else:
                st.warning('Url cannot parse correctly.')
        except:
            st.warning('Url cannot parse correctly.')
