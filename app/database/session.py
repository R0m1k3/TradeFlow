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


def init_database() -> None:
    """
    Initialize the database by creating all tables if they don't exist.
    Safe to call multiple times (idempotent).
    """
    Base.metadata.create_all(_engine)


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
