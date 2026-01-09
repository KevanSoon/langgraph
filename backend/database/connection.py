from config.database import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_PASS, POSTGRES_USER, CONNECTION_STRING
from psycopg_pool import AsyncConnectionPool
import psycopg2

def create_async_pool():
    """Create AsyncConnectionPool with proper settings"""
    return AsyncConnectionPool(
        conninfo=CONNECTION_STRING,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": None,
        }
    )


def connect_to_postgres():
    """Connect to Supabase PostgreSQL database (sync)"""
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASS
    )
    return conn