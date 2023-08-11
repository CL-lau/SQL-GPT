# import logger.logger as logging
import logging
import re
import time

import torch
import torch.nn as nn
from sqlalchemy import create_engine, text, inspect

from sql.SQL_type import get_db_operation_class, SQL_class


# from sql.SQL_class import


class SQL_operator(nn.Module):
    def __init__(self, jdbcList=None, userNames=None, passwords=None):
        super().__init__()
        self.jdbcList = jdbcList
        if self.jdbcList is None or len(self.jdbcList) <= 0:
            self.jdbcList = []
        self.userNames = userNames
        self.passwords = passwords
        self.tableList = {}
        self.columnsList = {}
        self.connMap = {}
        self.get_table_columns()
        self.sql_class = SQL_class()

    def get_table_columns(self):
        for jdbc in self.jdbcList:
            # self.connMap[jdbc] = create_engine('mysql+pymysql://root:password@localhost/test_db')
            self.connMap[jdbc] = create_engine('mysql+pymysql://' + self.userNames + ':' +
                                               self.passwords + '@' + jdbc)
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
