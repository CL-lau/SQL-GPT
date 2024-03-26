# import streamlit as st
#
# # Store the initial value of widgets in session state
# if "visibility" not in st.session_state:
#     st.session_state.visibility = "visible"
#     st.session_state.disabled = False
#
# col1, col2 = st.columns(2)
#
# with col1:
#     st.checkbox("Disable selectbox widget", key="disabled")
#     st.radio(
#         "Set selectbox label visibility 👉",
#         key="visibility",
#         options=["visible", "hidden", "collapsed"],
#     )
#
# with col2:
#     option = st.selectbox(
#         "How would you like to be contacted?",
#         ("Email", "Home phone", "Mobile phone"),
#         label_visibility=st.session_state.visibility,
#         disabled=st.session_state.disabled,
#     )
import subprocess
import sys

# import pymysql
# import pandas as pd
# db = pymysql.connect(host='localhost', port=3306, user="root", password="root", database='differ')
# cursor = db.cursor()
# sql = "SELECT * FROM change_dto"  # 替换为你的SQL语句
# cursor.execute(sql)
# results = cursor.fetchall()
# cursor.close()
# db.close()
# dataframe = pd.DataFrame(results, columns=[i[0] for i in cursor.description])  # 根据你的列名进行修改
# print(dataframe.head())


subprocess.run(['streamlit', 'run', "C:\\Users\\liuc8\\Desktop\\project\\SQL-GPT\\app.py"] + sys.argv[1:])
