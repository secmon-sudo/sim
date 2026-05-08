"""
SIM — Supabase/PostgreSQL Client
Blueprint V20.1 §5.1

Thread-safe connection pool for pipeline and Streamlit.
"""

import logging
import os

import psycopg
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

# Global pool reference for non-Streamlit environments
_pool = None

def _build_database_url() -> str:
    """Build database URL from env or secrets."""
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        try:
            import streamlit as st
            if "DATABASE_URL" in st.secrets:
                database_url = st.secrets["DATABASE_URL"]
            elif "database" in st.secrets and "url" in st.secrets["database"]:
                database_url = st.secrets["database"]["url"]
        except Exception:
            pass

    if not database_url:
        host = os.environ.get("SUPABASE_DB_HOST", "localhost")
        port = os.environ.get("SUPABASE_DB_PORT", "5432")
        dbname = os.environ.get("SUPABASE_DB_NAME", "postgres")
        user = os.environ.get("SUPABASE_DB_USER", "postgres")
        password = os.environ.get("SUPABASE_DB_PASSWORD", "")
        
        # Auto-switch to Transaction Mode (6543) if using Supabase Pooler on 5432
        if "pooler.supabase.com" in host and port == "5432":
            logger.info("Supabase pooler detected on 5432; auto-switching to port 6543 (Transaction Mode)")
            port = "6543"
            
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    # If the database_url itself contains the pooler host on 5432, fix it
    if "pooler.supabase.com" in database_url and ":5432/" in database_url:
        logger.info("Supabase pooler detected in URL on 5432; auto-switching to port 6543 (Transaction Mode)")
        database_url = database_url.replace(":5432/", ":6543/")

    return database_url

def get_pool() -> ConnectionPool:
    """Get or create a thread-safe connection pool."""
    # If running inside Streamlit, use st.cache_resource to persist the pool across reruns
    try:
        import streamlit as st
        return _get_streamlit_pool()
    except (ImportError, RuntimeError):
        # Fallback for pipeline/CLI
        global _pool
        if _pool is None or _pool.closed:
            database_url = _build_database_url()
            _pool = ConnectionPool(
                database_url,
                min_size=1,
                max_size=20,
                open=True,
                kwargs={"prepare_threshold": 0},  # Disable prepared statements for Supabase pooler
            )
            logger.info("Database connection pool created (max_size=20, prepare_threshold=0)")
        return _pool

def _get_streamlit_pool():
    """Streamlit-cached pool creation."""
    import streamlit as st
    
    # We use a nested function to allow st.cache_resource to work correctly
    @st.cache_resource(ttl=3600)
    def _create_pool():
        database_url = _build_database_url()
        # Basic validation: if host is localhost but we're likely in cloud, warn
        if "localhost" in database_url and os.environ.get("STREAMLIT_RUNTIME_ENV"):
             logger.warning("Database URL points to localhost in a Streamlit environment. Check secrets.")
             
        pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=20,
            open=True,
            kwargs={"prepare_threshold": 0},  # Disable prepared statements for Supabase pooler
        )
        logger.info("Streamlit shared database pool created (max_size=20, prepare_threshold=0)")
        return pool
        
    return _create_pool()


def get_connection():
    """Get a connection from the pool with a clear error if it fails."""
    try:
        pool = get_pool()
        # We use a 10s timeout instead of 30s for faster feedback in UI
        return pool.getconn(timeout=10.0)
    except Exception as e:
        if "timeout" in str(e).lower():
            raise RuntimeError("Database pool exhausted (10s timeout). Too many concurrent users or slow queries.") from e
        raise e


def put_connection(conn):
    """Return a connection to the pool."""
    pool = get_pool()
    pool.putconn(conn)


def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool is not None and not _pool.closed:
        _pool.close()
        logger.info("Database connection pool closed")
    _pool = None
