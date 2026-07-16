"""
SIM — Supabase/PostgreSQL Client
Blueprint V20.1 §5.1

Thread-safe connection pool for the pipeline and CLI runs.
"""

import logging
import os

import psycopg
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_pool = None

def _build_database_url() -> str:
    """Build database URL from environment variables."""
    database_url = os.environ.get("DATABASE_URL")

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
    global _pool
    if _pool is None or _pool.closed:
        database_url = _build_database_url()
        _pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=20,
            open=True,
            # check: don't hand out a pooled connection that died while idle.
            check=ConnectionPool.check_connection,
            kwargs={
                "prepare_threshold": None,  # Disable prepared statements for Supabase pooler
                # TCP keepalives: runner↔Supabase connections die silently mid-run
                # (half-open TCP); without probes the client blocks in wait() until
                # the server's 900s idle-in-transaction reaper kills the session
                # (failures of 2026-07-15 22:45 and 2026-07-16 06:00). Probe after
                # 30s idle, every 10s, 3 misses → dead in ~1 min instead of 15.
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 3,
            },
        )
        logger.info("Database connection pool created (max_size=20, prepare_threshold=None, tcp_keepalives=on)")
    return _pool


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
