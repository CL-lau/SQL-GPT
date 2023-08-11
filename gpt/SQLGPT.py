from gpt.chat import ChatGPT
from sql.SQL_operator import SQL_operator
from sql.SQL_type import get_db_operation_class, SQL_class
from sql.orm import orm


class SQL_GPT(ChatGPT):
    def __init__(self, ):
        super().__init__(OPENAI_API_KEY="", OPENAI_API_BASE="")
        self.SQL_operator = SQL_operator()
        self.orm = orm()
        self.SQL_prompt = "你现在需要扮演一个SQL命令只能生成器，需要根据我们输入的SQL描述来生成具体的SQL命令。" \
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
        self.error_format = "输入的sql语句为{}, 出现的错误是{}，关联的数据库的结构为{}。"
        self.question_init = "输出时，只需要给出具体的SQL命令，一定不要包含任其他任何问题也不需要解释以及其他任何文字。当前的MYSQL版本为{}"

    def generateSQL(self, SQL_question, need_operate=False, only_sql=True):
        questions = []
        if only_sql:
            questions.append(self.question_init.format(""))
        questions.append(SQL_question)
        sql = self.chat(questions=questions, system_assistant=self.SQL_prompt)
        sql = self.processSQL(sql)
        # 获取具体的SQL类型
        operator_type = get_db_operation_class(sql)

        if need_operate:
            if operator_type == SQL_class.SELECT or operator_type == SQL_class.INSERT:
                result = self.SQL_operator.operate_SQL(sql)
                return sql, result
        return sql

    def SQL_ERROR_CHECK(self, sql, error, need_operate=False, only_sql=True):
        question = self.error_format.format(sql, error, "")
        error_questions = []
        if only_sql:
            error_questions.append(self.question_init.format(""))
        error_questions.append(question)
        sql = self.chat(questions=error_questions, system_assistant=self.SQL_ERROR_CHECK_prompt)
        if need_operate:
            result = self.SQL_operator.operate_SQL(sql)
            return sql, result
        return sql

    def processSQL(self, sql):
        if str(sql).__contains__('\n'):
            start = str(sql).find('\n') + 1
            end = str(sql).rfind('\n')
            sql = sql[start: end]
            sql = str(sql).replace('\n', '')
            print(self.orm.data_file)
        return sql
