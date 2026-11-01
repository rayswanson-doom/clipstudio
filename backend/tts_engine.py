"""
backend/tts_engine.py

Text-to-speech with the open-source `edge-tts` library: free, no API key.

NOTE: edge-tts is not an offline engine. It streams neural-voice audio from
Microsoft's Edge "read aloud" service, so an internet connection is required
while generating. The resulting file is saved on your machine and can be used
offline afterwards.

Install:
  pip install edge-tts

Usage:
  import asyncio
  from tts_engine import generate_voiceover
  asyncio.run(generate_voiceover("Hello world", "out/voice.mp3"))
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "en-US-AriaNeural"


async def generate_voiceover(
    text: str,
    output_path: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
) -> str:
    """Synthesize `text` to an MP3 at `output_path` and return that path.

    rate/volume/pitch use edge-tts syntax, e.g. rate="+10%", pitch="-5Hz".
    The file is written to a temporary name and renamed on success, so a failed
    run never leaves a half-written voiceover behind.

    Raises ValueError for empty text or an empty output path, and RuntimeError
    if synthesis fails (e.g. no internet or an unknown voice).
    """
    if not text or not text.strip():
        raise ValueError("text must be non-empty")
    if not output_path or not str(output_path).strip():
        raise ValueError("output_path must be non-empty")

    import edge_tts  # lazy import so the module loads without the package

    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")

    try:
        communicate = edge_tts.Communicate(
            text.strip(), voice, rate=rate, volume=volume, pitch=pitch
        )
        await communicate.save(str(tmp))
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError("edge-tts produced no audio")
        os.replace(tmp, out)
    except Exception as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RuntimeError(f"Voiceover generation failed: {exc}") from exc

    logger.info("Voiceover saved to %s", out)
    return str(out.resolve())


async def list_voices(locale_prefix: str = "en") -> List[str]:
    """Return available voice short names whose locale starts with the prefix."""
    import edge_tts

    voices = await edge_tts.list_voices()
    return sorted(v["ShortName"] for v in voices if v["Locale"].lower().startswith(locale_prefix.lower()))


def generate_voiceover_sync(text: str, output_path: str, **kwargs) -> str:
    """Blocking wrapper for callers that are not async."""
    return asyncio.run(generate_voiceover(text, output_path, **kwargs))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Generate a voiceover MP3")
    ap.add_argument("text")
    ap.add_argument("output_path")
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    a = ap.parse_args()
    print(generate_voiceover_sync(a.text, a.output_path, voice=a.voice))
