"""User settings persistence (thresholds, grace period, baseline, etc.)."""

import json
import os
import threading

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULTS = {
    "baseline_angle": None,
    "slouch_threshold_degrees": 5,
    "grace_period_seconds": 3,
    "alert_cooldown_seconds": 0,
    "camera_index": 0,
}

_lock = threading.Lock()


class Config:
    """Thin wrapper around the config.json dict with load/save helpers."""

    def __init__(self, path: str = CONFIG_PATH):
        self.path = path
        self.data = dict(DEFAULTS)
        self.load()

    def load(self) -> "Config":
        with _lock:
            if os.path.exists(self.path):
                try:
                    with open(self.path, "r") as f:
                        loaded = json.load(f)
                    self.data = {**DEFAULTS, **loaded}
                except (json.JSONDecodeError, OSError):
                    self.data = dict(DEFAULTS)
        return self

    def save(self) -> None:
        with _lock:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)

    @property
    def is_calibrated(self) -> bool:
        return self.data.get("baseline_angle") is not None

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)

    def update(self, **kwargs) -> None:
        self.data.update(kwargs)
        self.save()
