import json
import logging
import os
import re
import time

import torch
import torch.nn as nn
from sqlalchemy import create_engine, text, inspect, MetaData, Table

from sql.SQL_type import get_db_operation_class, SQL_class


class SQL_operator(nn.Module):
    def __init__(self, jdbcList=None, userNames=None, passwords=None):
        super().__init__()

        self.jdbcList = jdbcList  # jdbcList的每一项的格式[host/db_name]
        if self.jdbcList is None or len(self.jdbcList) <= 0:
            self.jdbcList = []
        self.userNames = userNames
        self.userNameMap = {}
        if self.userNames is None or len(self.userNames) <= 0:
            self.userNames = []
            self.userNameMap = {}
        self.passwords = passwords
        self.userNameMap = {}
        if self.passwords is None or len(self.passwords) <= 0:
            self.passwords = []
            self.passwordMap = {}
        self.tableList = {}
        self.columnsList = {}
        self.connMap = {}

        self.config_file = "config.json"
        self.initJDBC()
        self.get_table_columns()
        self.sql_class = SQL_class()

    def initJDBC(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as config_file:
                config = json.load(config_file)
            jdbc_s = config['jdbc']
            for jdbc in jdbc_s:
                host = jdbc['host']
                db = jdbc['db']
                name = jdbc['name']
                password = jdbc['password']
                self.jdbcList.append(host + '/' + db)
                self.userNames.append(name)
                self.passwords.append(password)
                self.userNameMap[host + '/' + db] = name
                self.passwordMap[host + '/' + db] = password
                self.connMap[host + '/' + db] = create_engine('mysql+pymysql://' + name + ':' + password + '@' + host + '/' + db)

    def get_table_columns(self):
        for jdbc in self.jdbcList:
            # self.connMap[jdbc] = create_engine('mysql+pymysql://root:password@localhost/test_db')
            # self.connMap[jdbc] = create_engine('mysql+pymysql://' + self.userNames + ':' +
            #                                    self.passwords + '@' + jdbc)
            inspector = inspect(self.connMap[jdbc])
            tables = inspector.get_table_names()
            self.tableList[jdbc] = [table for table in tables]
            for table in tables:
                self.columnsList[jdbc + "_" + table] = [column for column in inspector.get_columns(table)]

    def process_result(self, sql, result):
        operator_type = get_db_operation_class(sql)
        if operator_type == self.sql_class.INSERT:
            pass
        elif operator_type == self.sql_class.SELECT:
            for col in result:
                print(col)
                print('\n')
            pass
        elif operator_type == self.sql_class.UPDATE:
            pass
        elif operator_type == self.sql_class.ALTER:
            pass
        elif operator_type == self.sql_class.CREATE:
            pass
        elif operator_type == self.sql_class.DELETE:
            pass
        elif operator_type == self.sql_class.DROP:
            pass
        elif operator_type == self.sql_class.TRANSACTION:
            pass
        else:
            logging.error("process operate error, the operate type not found.")

        return result

    def operate_SQL(self, sql):
        # SQL包含表名以及数据库名称
        match = re.search(r'FROM\s+(\S+)\.(\S+)', sql)
        database_name = None
        table_name = None
        if match:
            database_name = match.group(0)
            table_name = match.group(1)
        # SQL不包含数据库名，只包含表名
        match = re.search(r'FROM\s+(\S+)', sql)
        if match:
            table_name = match.group(1)
        else:
            logging.error("Table Name is not found, please check SQL")
        if database_name is None:
            for jdbc in self.jdbcList:
                tables = self.tableList.get(jdbc)
                if tables.__contains__(table_name):
                    database_name = str(jdbc).split('/')[-1]
                    break
        if database_name is None:
            if table_name is None:
                logging.error("Cannot find the database and table")
            else:
                logging.error("Cannot find the database contain table: " + table_name)
        # 执行结果
        if database_name is not None or table_name is not None:
            for jdbc in self.jdbcList:
                if str(jdbc).split('/')[-1] == database_name:
                    engine = self.connMap.get(jdbc)
                    with engine.connect() as conn:
                        start_time = time.time()
                        result = conn.execute(text(sql))
                        end_time = time.time()
                    logging.info("sql: " + sql + " cost time " + str(end_time - start_time))
        result = self.process_result(sql, result)
        return result

    def get_db_structure_and_index(self, sql):
        """
        根据包含数据库和数据库表的SQL语句获取数据库结构和索引信息
        :param sql: 包含数据库和数据库表的SQL语句，例如：SELECT * FROM mydatabase.mytable
        :return: 数据库结构和索引信息
        """
        db_name, table_name, jdbc = self.extract_database_and_table_from_sql(sql)
        engine = create_engine('mysql+pymysql://' + self.userNameMap[jdbc] + ':' + self.passwordMap[jdbc] + '@' + jdbc)
        metadata = MetaData()
        metadata.bind = engine
        table_obj = Table(table_name, metadata, autoload=True)
        return table_obj.columns, table_obj.indexes

    def extract_database_and_table_from_sql(self, sql):
        """
        从SQL语句中提取数据库名称和表名称
        :param sql: SQL语句
        :return: 数据库名称和表名称, jdbc
        """
        pattern_with_database = r'FROM\s+`?(\w+)`?\.`?(\w+)`?'
        pattern_without_database = r'FROM\s+`?(\w+)`?'

        match_with_database = re.search(pattern_with_database, sql, re.IGNORECASE)
        match_without_database = re.search(pattern_without_database, sql, re.IGNORECASE)

        if match_with_database:
            db_name, table_name = match_with_database.group(1), match_with_database.group(2)
            for jdbc in self.jdbcList:
                if str(jdbc).split('/') == db_name:
                    return db_name, table_name, jdbc
        elif match_without_database:
            table_name = match_with_database.group(1)
            for jdbc in self.tableList.keys():
                if self.tableList[jdbc].__contains__(table_name):
                    return str(jdbc).split('/')[-1], match_without_database.group(1), jdbc

        return None, None, None

    def get_db_structure_by_jdbc(self, jdbc, table_name):
        """
        :param table_name:
        :param jdbc
        :return: 数据库结构信息
        """
        engine = self.connMap[jdbc]
        metadata = MetaData()
        metadata.bind = engine
        table_obj = Table(table_name, metadata, autoload=True)
        return table_obj.columns
