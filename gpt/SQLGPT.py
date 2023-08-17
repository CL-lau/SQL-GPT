import logging
import re

from gpt.chat import ChatGPT
from sql.SQL_operator import SQL_operator
from sql.SQL_type import get_db_operation_class, SQL_class
from sql.orm import orm


class SQL_GPT(ChatGPT):
    def __init__(self, context=False):
        super().__init__(OPENAI_API_KEY="", OPENAI_API_BASE="")
        self.SQL_operator = SQL_operator()
        self.orm = orm()
        self.context = context
        self.SQL_prompt = "你现在需要扮演一个SQL命令智能生成器，需要根据我们输入的SQL描述来生成具体的SQL命令。" \
                          "你只需要给出具体的SQL命令，不需要给出任何其他的内容，并且要保证给出的SQL命令的可执行性以及正确性。"
        self.SQL_ERROR_CHECK_prompt = "你现在需要扮演一个SQL命令修改器，根据下面提供的 SQL 语句和错误信息对原始的SQL语句进行修改。" \
                                      "我推荐您尝试以下修改：" \
                                      "检查 SQL 语句是否符合语法规范，特别是拼写错误、缺少关键字等问题。" \
                                      "检查表名、列名等是否正确，是否存在拼写错误或大小写问题。" \
                                      "检查 SQL 语句中的条件是否正确，是否存在逻辑错误或语义错误。" \
                                      "检查 SQL 语句中的函数、聚合函数等是否正确，是否存在参数错误或语法错误。" \
                                      "检查 SQL 语句中的连接条件是否正确，是否存在连接类型错误或连接条件错误。" \
                                      "检查 SQL 语句中的子查询是否正确，是否存在语法错误或语义错误。" \
                                      "检查 SQL 语句中的排序、分组等是否正确，是否存在语法错误或语义错误。" \
                                      "检查 SQL 语句中的数据类型是否正确，是否存在数据类型转换错误或数据类型不匹配问题。"
        self.SQL_optimize_prompt = "你现在需要扮演一个SQL命令优化器，根据下面提供的 SQL 语句和数据库的结构以及相关信息来进行优化。" \
                                   "索引优化：为您QL语句添加合适的索引，提高查询效率。" \
                                   "SQL语句重构：优化SQL语句的结构，减少查询时间"
        self.error_format = "输入的sql语句为{}, 出现的错误是{}，关联的数据库的结构为{}。"
        self.question_init = "输出时，只需要给出具体的SQL命令，一定不要包含任其他任何问题也不需要解释以及其他任何文字。当前的MYSQL版本为{}"
        self.db_structure_format = "输入的sql语句为{}，这条sql语句涉及到的数据库表的结构如下：{}，索引等信息如下{}"
        self.java_orm_prompt = "你现在需要扮演一个mybatis xml文件智能生成器，需要根据我们输入的SQL描述以及数据库表的结构来生成具体的xml文件。" \
                               "你只需要给出这个xml文件中的代码，不需要给出其他的任何内容，并且如果需要包含返回值，你还需要给出对应的resultMap。"
        self.java_orm_format = "输入的功能描述为{}，关联的数据库表结构为{}"
        self.xml_init = "输出时，只需要给出具体的代码，一定不要包含任其他任何问题也不需要解释以及其他任何文字。"

    def generateSQL(self, SQL_question, need_operate=False, only_sql=True):
        questions = []
        if only_sql:
            questions.append(self.question_init.format(""))
        questions.append(SQL_question)
        if not self.context:
            sql = self.chat(questions=questions, system_assistant=self.SQL_prompt)
        else:
            sql = self.contextual_chat(questions=questions, system_assistant=self.SQL_prompt)
        sql = self.processSQL(sql)

        # 仅针对SELECT和INSERT来进行获取结果的可能。
        if need_operate:
            # 获取具体的SQL类型
            operator_type = get_db_operation_class(sql)
            if operator_type == SQL_class.SELECT or operator_type == SQL_class.INSERT:
                result = self.SQL_operator.operate_SQL(sql)
                return sql, result
        return sql

    def SQL_ERROR_CHECK(self, sql, error, need_operate=False, only_sql=True):
        """
        :param sql:
        :param error:
        :param need_operate:
        :param only_sql: 是否只返回SQL
        :return:
        """
        question = self.error_format.format(sql, error, "")
        error_questions = []
        if only_sql:
            error_questions.append(self.question_init.format(""))
        error_questions.append(question)
        if not self.context:
            sql = self.chat(questions=error_questions, system_assistant=self.SQL_ERROR_CHECK_prompt)
        else:
            sql = self.contextual_chat(questions=error_questions, system_assistant=self.SQL_ERROR_CHECK_prompt)
        if need_operate:
            result = self.SQL_operator.operate_SQL(sql)
            return sql, result
        return sql

    def optimizeSQL(self, sql, only_sql=True):
        db_structure, db_index = self.SQL_operator.get_db_structure_and_index(sql)
        question = self.SQL_optimize_prompt.format(sql, db_structure, db_index)
        optimize_questions = [question]
        if only_sql:
            optimize_questions.append(self.question_init.format(""))
        if not self.context:
            sql = self.chat(questions=optimize_questions, system_assistant=self.SQL_optimize_prompt)
        else:
            sql = self.contextual_chat(questions=optimize_questions, system_assistant=self.SQL_optimize_prompt)
        return sql

    def javaSQL(self, SQL_question, jdbc=None, table=None, xml_only=True):
        if jdbc is None or table is None:
            logging.error("SQL generation for java must specify a specific database "
                          "in the format ’host/db_name‘ and table.")
            return ""
        structure = self.SQL_operator.get_db_structure_by_jdbc(jdbc=jdbc, table_name=table)
        java_questions = [self.java_orm_format.format(SQL_question, str(structure))]
        if xml_only:
            java_questions.append(self.xml_init)
        if not self.context:
            xml = self.chat(questions=java_questions, system_assistant=self.java_orm_prompt)
        else:
            xml = self.contextual_chat(questions=java_questions, system_assistant=self.java_orm_prompt)
        return xml

    def processSQL(self, sql):
        if str(sql).__contains__('\n'):
            sql = str(sql).replace('\n', ' ')
        self.orm.save(sql)
        res = ""

        # 匹配SQL语句
        pattern = r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b.*?(?=;|$)'
        sql_list = re.findall(pattern, sql, re.IGNORECASE | re.DOTALL)
        # 输出匹配结果
        for item in sql_list:
            res = res + item
            res = res + " "
        return sql
