"""Helpers pour persister et lire les signaux IA depuis la DB."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.database.models import AISignal
from app.database.session import get_session

logger = logging.getLogger(__name__)


def save_ai_signal(
    symbol: str,
    mode: str,
    computed_at: datetime,
    interval: str = "1d",
    score: float | None = None,
    action: str | None = None,
    confidence: float | None = None,
    position_size_pct: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    rationale: str = "",
    key_risks: str = "",
    sources: list | None = None,
) -> None:
    """Persiste un signal IA dans la base de donnees."""
    session = get_session()
    try:
        # Supprimer les anciens signaux pour ce symbol/interval/mode
        session.query(AISignal).filter(
            AISignal.symbol == symbol,
            AISignal.interval == interval,
            AISignal.mode == mode,
        ).delete()

        signal = AISignal(
            symbol=symbol,
            interval=interval,
            mode=mode,
            score=score,
            action=action,
            confidence=confidence,
            position_size_pct=position_size_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            rationale=rationale,
            key_risks=key_risks,
            sources_json=json.dumps(sources) if sources else None,
            computed_at=computed_at,
        )
        session.add(signal)
        session.commit()
        logger.debug("Persisted AI signal for %s: %s", symbol, action or score)
    except Exception as exc:
        session.rollback()
        logger.error("Failed to persist AI signal for %s: %s", symbol, exc)
    finally:
        session.close()


def get_latest_ai_signal(symbol: str, interval: str = "1d", max_age_hours: int = 2) -> dict | None:
    """Retourne le dernier signal IA pour ce ticker s'il est recent."""
    session = get_session()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        row = (
            session.query(AISignal)
            .filter_by(symbol=symbol, interval=interval)
            .filter(AISignal.computed_at >= cutoff)
            .order_by(AISignal.computed_at.desc())
            .first()
        )
        return row.to_dict() if row else None
    except Exception as exc:
        logger.error("Failed to read AI signal for %s: %s", symbol, exc)
        return None
    finally:
        session.close()


def get_all_latest_ai_signals(interval: str = "1d", max_age_hours: int = 2) -> dict[str, dict]:
    """Retourne tous les signaux IA recents sous forme de dict {symbol: signal}."""
    session = get_session()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        rows = (
            session.query(AISignal)
            .filter_by(interval=interval)
            .filter(AISignal.computed_at >= cutoff)
            .order_by(AISignal.computed_at.desc())
            .all()
        )
        # Garder le plus recent par symbol
        result = {}
        for row in rows:
            if row.symbol not in result:
                result[row.symbol] = row.to_dict()
        return result
    except Exception as exc:
        logger.error("Failed to read all AI signals: %s", exc)
        return {}
    finally:
        session.close()
