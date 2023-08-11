from enum import Enum


def get_db_operation_class(sql):
    sql = sql.strip().upper()
    if sql.startswith('SELECT'):
        return SQL_class.SELECT
    elif sql.startswith('INSERT'):
        return SQL_class.INSERT
    elif sql.startswith('UPDATE'):
        return SQL_class.UPDATE
    elif sql.startswith('DELETE'):
        return SQL_class.DELETE
    elif sql.startswith('CREATE'):
        return SQL_class.CREATE
    elif sql.startswith('ALTER'):
        return SQL_class.ALTER
    elif sql.startswith('DROP'):
        return SQL_class.DROP
    elif sql.startswith('BEGIN') or sql.startswith('COMMIT') or sql.startswith('ROLLBACK'):
        return SQL_class.TRANSACTION
    else:
        raise ValueError('Unknown SQL operation')


class SQL_class:
    INSERT_STR = "INSERT"
    SELECT_STR = "SELECT"
    UPDATE_STR = "UPDATE"
    DELETE_STR = "DELETE"
    CREATE_STR = "CREATE"
    ALTER_STR = "ALTER"
    DROP_STR = "DROP"
    TRANSACTION_STR = "TRANSACTION"

    SELECT = 1
    INSERT = 2
    UPDATE = 3
    DELETE = 4
    CREATE = 5
    ALTER = 6
    DROP = 7
    TRANSACTION = 8

