"""Kill switch — external emergency stop.

Creating the file `data/KILL_SWITCH` on the host immediately halts all trading.
The bot checks for this file at the start of every tick.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default location relative to project data directory
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "KILL_SWITCH"


class KillSwitch:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_PATH

    @property
    def is_active(self) -> bool:
        return self._path.exists()

    @property
    def reason(self) -> str:
        if not self.is_active:
            return ""
        try:
            return self._path.read_text(encoding="utf-8").strip()
        except Exception:
            return "kill-switch file present"

    def activate(self, reason: str = "manual halt") -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(reason, encoding="utf-8")
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def deactivate(self) -> None:
        if self._path.exists():
            self._path.unlink()
            logger.warning("Kill switch deactivated")
