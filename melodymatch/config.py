"""Project-wide configuration values for MelodyMatch."""

from __future__ import annotations

DEFAULT_SONG_IDS = ("001", "002", "003", "004", "005")

DEFAULT_DATASET_ROOT_CANDIDATES = ("data/POP909",)

# The weights are normalized before scoring, so they do not need to sum to 1.
# Intervals are the strongest cue, followed by duration, contour, and beat phase.
DEFAULT_SIMILARITY_WEIGHTS = {
    "interval": 0.40,
    "duration": 0.30,
    "contour": 0.20,
    "beat": 0.10,
}


def normalize_song_id(song_id: str | int) -> str:
    """Return a POP909-style song id such as 001."""
    text = str(song_id).strip()
    return text.zfill(3) if text.isdigit() else text


def default_dataset_root() -> str:
    """Return the first common POP909 dataset path that exists."""
    from pathlib import Path

    for candidate in DEFAULT_DATASET_ROOT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return DEFAULT_DATASET_ROOT_CANDIDATES[0]
