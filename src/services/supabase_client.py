"""
SIM — Supabase/PostgreSQL Client
Blueprint V20.1 §5.1

Connection pool singleton for pipeline and Streamlit.
"""

import logging
import os

import psycopg

logger = logging.getLogger(__name__)

_connection = None


def get_connection():
    """Get or create a database connection.

    Uses DATABASE_URL environment variable.
    Falls back to individual SUPABASE_* vars if DATABASE_URL is not set.
    """
    global _connection

    if _connection is not None and not _connection.closed:
        return _connection

    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        # Build from Supabase components
        host = os.environ.get("SUPABASE_DB_HOST", "localhost")
        port = os.environ.get("SUPABASE_DB_PORT", "5432")
        dbname = os.environ.get("SUPABASE_DB_NAME", "postgres")
        user = os.environ.get("SUPABASE_DB_USER", "postgres")
        password = os.environ.get("SUPABASE_DB_PASSWORD", "")
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    try:
        _connection = psycopg.connect(database_url, autocommit=False)
        logger.info("Database connection established")
        return _connection
    except Exception:
        logger.exception("Failed to connect to database")
        raise


def close_connection():
    """Close the database connection if open."""
    global _connection
    if _connection is not None and not _connection.closed:
        _connection.close()
        logger.info("Database connection closed")
    _connection = None
