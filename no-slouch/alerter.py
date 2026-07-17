"""Alert triggering, cooldown management, and session/alert logging."""

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from detector import PostureState

SESSION_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_log.json")

NOTIFICATION_TITLE = "Posture Check"
NOTIFICATION_MESSAGE = "You've been slouching -- sit up straight!"


class SessionLog:
    """Tracks good/slouch time and alerts for the current session, and
    appends a summary entry to session_log.json on finalize."""

    def __init__(self, path: str = SESSION_LOG_PATH):
        self.path = path
        self.session_start = time.time()
        self.alert_timestamps: List[str] = []
        self.good_seconds = 0.0
        self.slouch_seconds = 0.0

    def record_tick(self, state: PostureState, dt: float) -> None:
        if state == PostureState.GOOD:
            self.good_seconds += dt
        elif state == PostureState.SLOUCHING:
            self.slouch_seconds += dt
        # NO_PERSON: paused, doesn't count toward either bucket.

    def record_alert(self) -> None:
        self.alert_timestamps.append(datetime.now().isoformat(timespec="seconds"))

    def summary(self) -> dict:
        duration = time.time() - self.session_start
        tracked = self.good_seconds + self.slouch_seconds
        good_pct = (self.good_seconds / tracked * 100.0) if tracked > 0 else 0.0
        return {
            "started_at": datetime.fromtimestamp(self.session_start).isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": round(duration, 1),
            "alert_count": len(self.alert_timestamps),
            "alert_timestamps": self.alert_timestamps,
            "good_posture_percent": round(good_pct, 1),
        }

    def finalize_and_save(self) -> dict:
        summary = self.summary()
        entries = []
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    entries = json.load(f)
                if not isinstance(entries, list):
                    entries = []
            except (json.JSONDecodeError, OSError):
                entries = []
        entries.append(summary)
        with open(self.path, "w") as f:
            json.dump(entries, f, indent=2)
        return summary


class Alerter:
    """Fires system notifications + an audio chime, gated by a cooldown."""

    def __init__(self, cooldown_seconds: float = 300.0, session_log: Optional[SessionLog] = None):
        self.cooldown_seconds = cooldown_seconds
        self.session_log = session_log
        self._last_alert_time: Optional[float] = None

    def in_cooldown(self, now: Optional[float] = None) -> bool:
        if self._last_alert_time is None:
            return False
        now = now if now is not None else time.monotonic()
        return (now - self._last_alert_time) < self.cooldown_seconds

    def seconds_until_ready(self, now: Optional[float] = None) -> float:
        if self._last_alert_time is None:
            return 0.0
        now = now if now is not None else time.monotonic()
        remaining = self.cooldown_seconds - (now - self._last_alert_time)
        return max(0.0, remaining)

    def fire(self, now: Optional[float] = None) -> bool:
        """Attempt to fire an alert. Returns True if it actually fired
        (i.e. wasn't suppressed by an active cooldown)."""
        now = now if now is not None else time.monotonic()
        if self.in_cooldown(now):
            return False

        self._last_alert_time = now
        self._notify()
        self._play_chime()
        if self.session_log is not None:
            self.session_log.record_alert()
        return True

    @staticmethod
    def _notify() -> None:
        try:
            from plyer import notification

            notification.notify(
                title=NOTIFICATION_TITLE,
                message=NOTIFICATION_MESSAGE,
                app_name="Posture Correction",
                timeout=8,
            )
        except Exception as exc:  # pragma: no cover - platform dependent
            print(f"[alerter] notification failed ({exc}); {NOTIFICATION_MESSAGE}")

    @staticmethod
    def _play_chime() -> None:
        """Play a short audio chime without blocking the caller."""
        threading.Thread(target=Alerter._play_chime_blocking, daemon=True).start()

    @staticmethod
    def _play_chime_blocking() -> None:
        try:
            if sys.platform == "darwin":
                subprocess.run(
                    ["afplay", "/System/Library/Sounds/Ping.aiff"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif sys.platform.startswith("win"):
                import winsound

                winsound.Beep(880, 300)
            else:
                from playsound import playsound

                playsound("/usr/share/sounds/alsa/Front_Center.wav")
        except Exception as exc:  # pragma: no cover - platform dependent
            print(f"[alerter] chime failed ({exc})")
