import os
from dotenv import load_dotenv

load_dotenv()

# Postgres connection info
POSTGRES_HOST = os.getenv("SUPABASE_DB_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("SUPABASE_DB_PORT", 5432))
POSTGRES_DB = os.getenv("SUPABASE_DB_NAME", "postgres")
POSTGRES_USER = os.getenv("SUPABASE_DB_USER", "postgres")
POSTGRES_PASS = os.getenv("SUPABASE_DB_PASSWORD", "")
POSTGRES_SSLMODE = os.getenv("SUPABASE_DB_SSLMODE", "disable")  # Use "require" for Supabase

EMBED_DIMENSION = 1024


# Connection string for AsyncConnectionPool
CONNECTION_STRING = (
    f"postgres://{POSTGRES_USER}:{POSTGRES_PASS}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    f"?sslmode={POSTGRES_SSLMODE}"
)