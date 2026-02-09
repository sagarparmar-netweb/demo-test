import mysql.connector
from contextlib import contextmanager
from config.db_config import DBConfig


@contextmanager
def mysql_connection():
    connection = mysql.connector.connect(**DBConfig.get_config())
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
