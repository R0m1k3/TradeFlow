"""
TradeFlow — Database Session Manager
Provides SQLite connection and session factory via SQLAlchemy.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import Base


def _load_db_path() -> str:
    """
    Resolve the SQLite database file path from config.yaml or environment variable.

    Priority: ENV var DATABASE_PATH > config.yaml > default 'data/tradeflow.db'

    Returns:
        Absolute path string to the SQLite file.
    """
    env_path = os.environ.get("DATABASE_PATH")
    if env_path:
        return env_path

    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        db_relative = config.get("database", {}).get("path", "data/tradeflow.db")
        # Resolve relative to project root (2 levels above app/)
        project_root = Path(__file__).resolve().parents[2]
        return str(project_root / db_relative)

    return str(Path("data/tradeflow.db").resolve())


def create_db_engine() -> Engine:
    """
    Create and return a SQLAlchemy engine connected to the SQLite database.
    Ensures the parent directory exists before creating the engine.

    Returns:
        Configured SQLAlchemy Engine instance.
    """
    db_path = _load_db_path()
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    return engine


# Module-level engine and session factory (singleton pattern)
_engine: Engine = create_db_engine()
_SessionFactory: sessionmaker = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def _migrate(engine: Engine) -> None:
    """Apply incremental ALTER TABLE migrations for columns added after initial release."""
    migrations = [
        # v2: live trading columns on sim_runs
        ("sim_runs", "is_live",      "INTEGER NOT NULL DEFAULT 0"),
        ("sim_runs", "status",       "VARCHAR(16) NOT NULL DEFAULT 'completed'"),
        ("sim_runs", "last_tick_at", "DATETIME"),
        # v2: reason column on trades
        ("trades",   "reason",       "TEXT DEFAULT ''"),
    ]
    with engine.connect() as conn:
        for table, column, col_def in migrations:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"
                    )
                )
                conn.commit()
            except Exception:
                # Column already exists — ignore
                pass

        # v3: composite index on price_cache for fast lookups
        try:
            conn.execute(
                __import__("sqlalchemy").text(
                    "CREATE INDEX IF NOT EXISTS ix_price_cache_sym_int_ts "
                    "ON price_cache (symbol, interval, timestamp)"
                )
            )
            conn.commit()
        except Exception:
            pass

        # v4: ai_signals and bot_decisions tables
        try:
            conn.execute(__import__("sqlalchemy").text(
                """CREATE TABLE IF NOT EXISTS ai_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol VARCHAR(16) NOT NULL,
                    interval VARCHAR(8) NOT NULL DEFAULT '1d',
                    mode VARCHAR(16) NOT NULL DEFAULT 'autonomous',
                    score FLOAT,
                    action VARCHAR(8),
                    confidence FLOAT,
                    position_size_pct FLOAT,
                    stop_loss_pct FLOAT,
                    take_profit_pct FLOAT,
                    rationale TEXT,
                    key_risks TEXT,
                    sources_json TEXT,
                    computed_at DATETIME NOT NULL
                )"""
            ))
            conn.execute(__import__("sqlalchemy").text(
                "CREATE INDEX IF NOT EXISTS ix_ai_sym_int_ts "
                "ON ai_signals (symbol, interval, computed_at)"
            ))
            conn.execute(__import__("sqlalchemy").text(
                """CREATE TABLE IF NOT EXISTS bot_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sim_run_id INTEGER NOT NULL REFERENCES sim_runs(id),
                    symbol VARCHAR(16) NOT NULL,
                    timestamp DATETIME NOT NULL,
                    action VARCHAR(8) NOT NULL,
                    reason TEXT NOT NULL,
                    price FLOAT,
                    ai_action VARCHAR(8),
                    ai_confidence FLOAT
                )"""
            ))
            conn.commit()
        except Exception:
            pass


def init_database() -> None:
    """
    Initialize the database by creating all tables if they don't exist,
    then apply any pending column migrations. Safe to call multiple times.
    """
    Base.metadata.create_all(_engine)
    _migrate(_engine)


def get_session() -> Session:
    """
    Return a new SQLAlchemy database session.
    Caller is responsible for committing and closing the session.

    Returns:
        A new SQLAlchemy Session instance.

    Example:
        session = get_session()
        try:
            session.add(obj)
            session.commit()
        except Exception as exc:
            session.rollback()
            raise
        finally:
            session.close()
    """
    return _SessionFactory()
