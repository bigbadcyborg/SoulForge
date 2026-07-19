"""Qt-free helpers for the GUI (safe to import without PySide6).

Pure logic — rectangle math, screen capture, hotkey-string conversion — kept
here so it can be unit-tested on any platform without a display.
"""

from __future__ import annotations

import io


def normalize_rect(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
    """Return (left, top, width, height) from two drag corners, unordered."""
    left = min(x0, x1)
    top = min(y0, y1)
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    return left, top, width, height


def capture_region_png(left: int, top: int, width: int, height: int) -> bytes:
    """Grab a screen region as PNG bytes using mss + Pillow (Windows GUI venv)."""
    import mss
    from PIL import Image

    with mss.mss() as sct:
        shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def command_matches(query: str, usage: str, description: str) -> bool:
    """Case-insensitive substring match of a help command against a search query."""
    q = query.strip().lower()
    if not q:
        return True
    return q in usage.lower() or q in description.lower()


def to_pynput_hotkey(hotkey: str) -> str:
    """Convert 'ctrl+alt+s' to pynput's '<ctrl>+<alt>+s' format."""
    mods = {"ctrl", "alt", "shift", "cmd", "super"}
    parts = [p.strip().lower() for p in hotkey.split("+") if p.strip()]
    return "+".join(f"<{p}>" if p in mods else p for p in parts)


def frames_to_wav_bytes(frames, sample_rate: int = 16000) -> bytes:
    """Encode recorded float32 mono frames (a list of numpy chunks) to WAV bytes.

    Whisper wants 16 kHz mono; the caller records at that rate. Kept separate
    from capture so the encoding is unit-testable with synthetic frames.
    """
    import numpy as np
    import soundfile as sf

    if not len(frames):
        audio = np.zeros(0, dtype="float32")
    else:
        audio = np.concatenate(frames).astype("float32").reshape(-1)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
