"""Small data models used by the backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MelodyNote:
    """A single symbolic note from the selected melody track."""

    pitch: int
    start: float
    end: float
    duration: float


@dataclass(frozen=True)
class MelodyFeatures:
    """Feature sequences and summary statistics for one song melody."""

    song_id: str
    midi_path: Path
    track_name: str
    fallback_used: bool
    beat_source: Path | None
    pitches: list[int]
    intervals: list[int]
    durations: list[float]
    contours: list[int]
    beat_positions: list[float]
    note_count: int
    average_pitch: float
    average_duration: float
    pitch_range: int

    def summary(self) -> dict[str, object]:
        """Return one-row-friendly metadata about the extracted melody."""
        return {
            "song_id": self.song_id,
            "midi_path": str(self.midi_path),
            "track_name": self.track_name,
            "fallback_used": self.fallback_used,
            "beat_source": str(self.beat_source) if self.beat_source else "",
            "note_count": self.note_count,
            "average_pitch": self.average_pitch,
            "average_duration": self.average_duration,
            "pitch_range": self.pitch_range,
        }

