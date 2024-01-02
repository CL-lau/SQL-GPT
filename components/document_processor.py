import os
import tempfile
from typing import Tuple

import streamlit as st

from model import DocumentLoader


def upload_and_process_document() -> tuple[int, list]:
    st.write('#### Upload a Document file')
    browse, url_link, sql, code_rep, ocr_ = st.tabs(
        ['  Document  ', '  Document Online  ', '  Mysql Analysis  ', '  Code Rep  ', '  OCR Image  ']
    )
    with browse:
        upload_file = st.file_uploader(
            'Browse file (.pdf, .docx, .csv, `.txt`)',
            type=['pdf', 'docx', 'csv', 'txt'],
            label_visibility='hidden'
        )
        filetype = os.path.splitext(upload_file.name)[1].lower() if upload_file else None
        upload_file_name = upload_file.name if upload_file else None
        upload_file = upload_file.read() if upload_file else None

    with url_link:
        doc_url = st.text_input(
            "Enter document URL Link (.pdf, .docx, .csv, .txt)",
            placeholder='https://www.xxx/uploads/file.pdf',
            label_visibility='hidden'
        )
        if doc_url:
            upload_file, filetype = DocumentLoader.crawl_file(doc_url)

    with sql:
        sql_url = st.text_input(
            "Enter Mysql URL Link.",
            placeholder='127.0.0.1:3306/database_name',
        )
        username, password = st.columns(2)
        with username:
            sql_username = st.text_input(
                "Enter Mysql UserName.",
                placeholder='UserName',
            )
        with password:
            sql_password = st.text_input(
                "Enter Mysql Password.",
                placeholder='Password',
                type='password'
            )

    with code_rep:
        code_url = st.text_input(
            "Enter document URL Link (.pdf, .docx, .csv, .txt)",
            placeholder='https://www.xxx/',
            label_visibility='hidden'
        )

    with ocr_:
        upload_images = st.file_uploader(
            'Browse file (.png, .jpg)',
            type=['png', 'jpg'],
            label_visibility='hidden',
            accept_multiple_files=True
        )
        import pytesseract
        from PIL import Image
        imagetype = os.path.splitext(upload_images[0].name)[1].lower() if upload_images else None
        imageFile = upload_images[0].read() if upload_images else None
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    if imageFile is not None and imagetype is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(imageFile)
        text = ""
        for upload_image in upload_images:
            if imageFile.strip():
                file_path = os.path.join("uploads", upload_image.name)
                with open(file_path, "wb") as f:
                    f.write(upload_image.getbuffer())
                st.success(f"已保存文件: {upload_image.name}")

            image = Image.open("./uploads/" + upload_image.name)
            text = pytesseract.image_to_string(image, lang='chi_sim')
        st.markdown(text)

    if upload_file and filetype and upload_file_name:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(upload_file)
        # temp_file_path = temp_file

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(upload_file)
        if upload_file.strip():
            file_path = os.path.join("uploads", upload_file_name)
            with open(file_path, "wb") as f:
                f.write(upload_file)
            st.success(f"已保存文件: {upload_file_name}")

        docs = DocumentLoader.load_documents(os.path.join("uploads", upload_file_name), filetype)
        docs = DocumentLoader.split_documents(
            docs, chunk_size=1000,
            chunk_overlap=200
        )

        temp_file.close()
        # if temp_file_path:
        #     os.remove(temp_file_path)

        return 1, [docs]

    if sql_url:
        return 2, [sql_url, sql_username, sql_password]

    if code_url:
        return 3, [code_url]

    if ocr_:
        return 4, []

