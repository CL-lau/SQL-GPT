
# SQL_GPT: SQL 生成工具

## 简介

SQL_GPT 是一款强大的工具，它能够通过简单的文字描述，自动生成符合要求的 SQL 查询语句。无论您是在快速生成复杂查询还是与数据库进行交互，SQL_GPT 都能够显著提升您的效率和工作流程。

## 功能列表

- [x] **自动生成 SQL 查询：** 只需简单的文字描述，工具将自动生成符合要求的 SQL 查询语句。
- [x] **错误修正建议：** 在查询存在错误时，工具会提供智能的修正建议，助您快速解决问题。
- [x] **数据库连接管理：** 轻松配置和管理多个数据库连接，直接在工具内执行生成的 SQL 查询。
- [x] **多数据库兼容：** 工具兼容多种主流数据库系统，适用于不同项目的需求。
- [x] **代理访问支持：** 针对特定场景，您可以通过系统代理来访问 GPT 服务。
- [x] **多 API KEY 轮询：** 您可以设置多个备选 ```API KEY``` 来访问 GPT，提升稳定性。
- [x] **SQL语句优化：** 通过GPT的能力根据数据库的结构进行SQL语句优化。
- [x] **Java持久层SQL语句生成：** 结合SQL以及数据库结构信息来自动生成Java持久层语句，如：```Mybatis```。
- [x] **多轮对话：** 通过多轮对话的方式来实现对生成SQL的不断优化。
- [ ] **数据自动可视化分析：** 在通过数据库操作完数据之后，通过对数据进行分析来展示数据的基础信息。

## 快速开始指南

要开始使用 SQL_GPT，只需按照以下简单步骤进行操作：

1. **安装所需依赖：** 确保您的环境中已安装 Python 3.x，并执行以下命令安装所需依赖包：

    ```bash
    pip install requirements.txt
    ```
2. **配置OPENAI数据：** 在 ```config.json```中配置您的```OPENAI-KEY```以及```BASE_URL```信息来方便同Chat交互，还可以通过```OPENAI-KEYS```列表来设置多个APP_KEY。

3. **配置数据库连接：** 在 ```config.json```中配置您的数据库连接信息，包括主机名、用户名、密码等，以便进行数据库交互。

4. **生成 SQL 查询：** 在工具的用户界面中，用自然语言描述您的查询需求。SQL_GPT 将会智能地生成相应的 SQL 查询语句。例如：
   
   ```python
   from gpt.SQLGPT import SQL_GPT
   sql_GPT = SQL_GPT()
   
   # 生成sql语句
   sql_GPT.generateSQL("生成两个数据库表的关联查询操作。")
   
   # 对错误的SQL进行修改
   sql_GPT.SQL_ERROR_CHECK("SELECT * FROM tableA WHERE user_id IN (SELECT user_id FROM tableB LIMIT 1000);", "SQL执行失败: (1235, This version of MySQL doesn't yet support 'LIMIT & IN/ALL/ANY/SOME subquery'")
   ```

## Star历史

>>>
[![Star History Chart](https://api.star-history.com/svg?repos=CL-lau/SQL-GPT&type=Date)](https://star-history.com/#CL-lau/SQL-GPT&Date)
