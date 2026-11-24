"""
backend/upload_limits.py

The maximum size of a video that can be uploaded in the browser: 500 MB or 1 GB (1024 MB).

Streamlit reads `server.maxUploadSize` (in MB) once, when the server starts, so the choice is saved in a small settings
file and applied by the launcher at the next start (`--upload-limit`, the CINEPULSE_MAX_UPLOAD_MB environment variable
and `.streamlit/config.toml` are honoured in that order). The app also refuses a bigger file itself, with a clear message.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

MB = 1024 * 1024
CHOICES = {"500 MB": 500, "1 GB (1024 MB)": 1024}
DEFAULT_MB = 1024
ENV_VAR = "CINEPULSE_MAX_UPLOAD_MB"
SETTINGS_NAME = "settings.json"


def settings_path() -> Path:
    """settings.json next to the exe when packaged, else in the project root."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent.parent
    return base / "storage" / SETTINGS_NAME


def normalize(value) -> int:
    """Snap any value to one of the allowed limits (500 or 1024); anything else becomes the default."""
    try:
        v = int(float(str(value).lower().replace("mb", "").replace("gb", "").strip()))
    except (TypeError, ValueError):
        return DEFAULT_MB
    return v if v in CHOICES.values() else DEFAULT_MB


def read_limit_mb(path: Optional[Path] = None, env: Optional[dict] = None) -> int:
    """The limit to use at start-up: environment variable > saved setting > 1024."""
    env = os.environ if env is None else env
    if env.get(ENV_VAR):
        return normalize(env[ENV_VAR])
    try:
        data = json.loads(Path(path or settings_path()).read_text(encoding="utf-8"))
        return normalize(data.get("max_upload_mb", DEFAULT_MB))
    except (OSError, ValueError, AttributeError):
        return DEFAULT_MB


def write_limit_mb(value, path: Optional[Path] = None) -> int:
    """Save the choice (kept to 500 / 1024) and return it. Other saved settings are preserved."""
    limit = normalize(value)
    p = Path(path or settings_path())
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data["max_upload_mb"] = limit
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return limit


def label_for(mb: int) -> str:
    return next((k for k, v in CHOICES.items() if v == mb), "1 GB (1024 MB)")


def too_big(size_bytes: Optional[int], limit_mb: int) -> bool:
    return bool(size_bytes) and size_bytes > normalize(limit_mb) * MB


def too_big_message(name: str, size_bytes: int, limit_mb: int) -> str:
    return (f"“{name}” is {size_bytes / MB:,.0f} MB, which is over the {label_for(normalize(limit_mb))} limit. "
            "Pick the 1 GB limit in the sidebar (applies after restarting the app) or use the "
            "“video already on this computer” box, which has no size limit.")
