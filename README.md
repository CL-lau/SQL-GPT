[//]: # (# SQL_GPT: Tools for interacting with SQL and files are implemented through LLMs.)

<p align="center">
<a href="https://github.com/CL-lau/SQL-GPT">
<img src="./assets/main.png" alt="Chroma logo">

[//]: # (<center><span style="font-family: Arial; font-size: 30px;">SQL-GPT</span></center>)
</a>
</p>

<p align="center">
    <b>Tools for interacting with SQL and files are implemented through LLMs.</b>. <br />
    The most complete access interface is encapsulated
</p>

<p align="center">

[//]: # (  <a href="https://discord.gg/MMeYNTmh3x" target="_blank">)
[//]: # (      <img src="https://img.shields.io/discord/1073293645303795742" alt="Discord">)
[//]: # (  </a> |)
[//]: # (  <a href="https://github.com/chroma-core/chroma/blob/master/LICENSE" target="_blank">)
[//]: # (      <img src="https://img.shields.io/static/v1?label=license&message=Apache 2.0&color=white" alt="License">)
[//]: # (  </a> |)
  <a href="https://github.com/CL-lau/SQL-GPT/blob/main/README-zh.md" target="_blank">
      Chinese Docs
  </a> |
  <a href="https://github.com/CL-lau/SQL-GPT/blob/main/README.md" target="_blank">
      English Docs
  </a>
</p>

## Introduction

SQLGPT is a powerful tool that can generate SQL queries that meet your requirements through simple text descriptions. Whether you need to quickly generate complex queries or interact with databases, SQL_GPT can significantly improve your efficiency and workflow.

## Feature List

-[x] Automatic SQL Query Generation: Simply describe your query in text, and the tool will automatically generate the SQL query that meets your requirements.
-[x] Error Correction Suggestions: When there are errors in your query, the tool will provide intelligent suggestions for fixing them, helping you quickly resolve issues.
-[x] Database Connection Management: Easily configure and manage multiple database connections, and execute generated SQL queries directly within the tool.
-[x] Multi-Database Compatibility: The tool is compatible with multiple mainstream database systems, suitable for the needs of different projects.
-[x] Proxy Access Support: For specific scenarios, you can access the GPT service through a system proxy.
-[x] Multi-API KEY Rotation: You can set multiple backup API keys to access GPT, improving stability.
-[x] SQL Statement Optimization: Optimize SQL statements based on the structure of the database using GPT's capabilities.
-[x] Java Persistence Layer SQL Statement Generation: Generate Java persistence layer statements, such as Mybatis, based on SQL and database structure information.
-[x] Multi-Turn Dialogue: Continuously optimize generated SQL through multi-turn dialogue.
-[x] File System Dialogue: Use vector databases to organize file system information and complete dialogue with the file system.
-[x] Cache Operations to Accelerate File Dialogue: Use various Redis data structures to accelerate access to the vector database, improving average lookup speed by 30%.
-[ ] Automatic Data Visualization Analysis: Analyze data and display basic information after completing database operations.

## Quick Start Guide

To start using SQL_GPT, simply follow these simple steps:

1. **Install Required Dependencies:** Make sure Python 3.x is installed in your environment and execute the following command to install the required dependencies:

    ```bash
    pip install requirements.txt
    ```

2. **Configure OPENAI**: Configure your ```OPENAI-KEY``` and ```BASE_URL``` information in ```config.json``` to facilitate interaction with Chat. You can also set multiple ```APP_KEYs``` through the ```OPENAI-KEYS``` list.

3. **Configure database connection**: Configure your database connection information, including hostname, username, password, etc., in ```config.json``` for database interaction.

4. **Generate SQL**: In the tool's user interface, describe your query requirements in natural language. SQL_GPT will intelligently generate the corresponding SQL query statement. For example:
   
   ```python
   from gpt.SQLGPT import SQL_GPT
   from gpt.FILEGPT import File_GPT
   sql_GPT = SQL_GPT()
   file_gpt = File_GPT()
   
   # 生成sql语句
   sql_GPT.generateSQL("Perform a join operation on two database tables.")
   # answer: 'SELECT * FROM table1 JOIN table2 ON table1.column_name = table2.column_name;'
   
   # 对错误的SQL进行修改
   sql_GPT.SQL_ERROR_CHECK("SELECT * FROM tableA WHERE user_id IN (SELECT user_id FROM tableB LIMIT 1000);", "SQL执行失败: (1235, This version of MySQL doesn't yet support 'LIMIT & IN/ALL/ANY/SOME subquery'")
   
   # 向文件进行提问。
   file_gpt.addFile("2307_01504.pdf", "./embedding")
   file_gpt.askFile("who is the auther?")
   # answer: 'The author of this work is Xiangguo Sun, along with co-authors Hong Cheng, Jia Li, Bo Liu, and Jihong Guan.'

   ```

## System Architecture

SQl-GPT is a local question-answering system constructed based on LLM. It can generate SQL statements according to requirements, optimize and correct SQL statements, and generate MyBatis XML files based on them. In addition, it can directly execute SQL statements and set multiple monitored databases.
On the basis of interacting with the database, it also interacts with the file system, increases the interaction context by introducing vector databases, and caches queries using Redis structures to improve query speed, while supporting multiple vector models.

<center><span style="font-family: Arial; font-size: 13px;">Interact with local databases and file systems for Q&A</span></center>

![系统架构](./assets/frame.png)
### Prerequisites
- ```redis``` Install Redis database, it is recommended to install it through Docker.
   ```bash
   docker run --restart=always -p 6379:6379 --name redis-docker -d redis:7.0.12  --requirepass admin
   ```
- ```api_key``` When interacting with the online LLM, you need to apply for the corresponding api-key.
- ```MySql``` Install the MySql database locally, and it is also recommended to use Docker for installation.

### Acknowledgement

This project is standing on the shoulders of giants and is not going to work without the open-source communities. Special thanks to the following projects for their excellent contribution to the AI industry:
- [FastChat](https://github.com/lm-sys/FastChat) for providing chat services
- [vicuna-13b](https://lmsys.org/blog/2023-03-30-vicuna/) as the base model
- [langchain](https://langchain.readthedocs.io/) tool chain
- [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT) universal plugin template
- [Hugging Face](https://huggingface.co/) for big model management
- [Chroma](https://github.com/chroma-core/chroma) for vector storage
- [Milvus](https://milvus.io/) for distributed vector storage
- [ChatGLM](https://github.com/THUDM/ChatGLM-6B) as the base model
- [llama_index](https://github.com/jerryjliu/llama_index) for enhancing database-related knowledge using [in-context learning](https://arxiv.org/abs/2301.00234) based on existing knowledge bases.


## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=CL-lau/SQL-GPT&type=Date)](https://star-history.com/#CL-lau/SQL-GPT&Date)