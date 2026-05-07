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
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    return database_url


def get_pool() -> ConnectionPool:
    """Get or create a thread-safe connection pool."""
    global _pool
    if _pool is None or _pool.closed:
        database_url = _build_database_url()
        _pool = ConnectionPool(
            database_url,
            min_size=2,
            max_size=20,
            open=True,
            configure=lambda conn: conn.execute("SET application_name = 'sim_dashboard'"),
        )
        logger.info("Database connection pool created (max_size=20)")
    return _pool


def get_connection(timeout: float = 60.0):
    """Get a connection from the pool (legacy single-use compatibility)."""
    pool = get_pool()
    return pool.getconn(timeout=timeout)


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
