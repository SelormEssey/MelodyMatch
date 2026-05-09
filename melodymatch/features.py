"""Feature extraction helpers for symbolic melody notes."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from statistics import mean

from melodymatch.models import MelodyFeatures, MelodyNote


def contour_from_interval(interval: int) -> int:
    """Map an interval to -1 for down, 0 for same, and 1 for up."""
    if interval > 0:
        return 1
    if interval < 0:
        return -1
    return 0


def beat_phase(note_start: float, beat_times: list[float]) -> float | None:
    """
    Return a note's phase inside the local beat interval.

    A value near 0 means the note starts on a beat. A value near 0.5 means it
    starts halfway between two beats. The value is circular during comparison,
    so 0.95 and 0.05 are considered close.
    """
    if len(beat_times) < 2:
        return None

    index = bisect_right(beat_times, note_start) - 1
    if index < 0:
        return 0.0

    if index >= len(beat_times) - 1:
        previous = beat_times[index - 1] if index > 0 else beat_times[index]
        interval = beat_times[index] - previous
        if interval <= 0:
            return None
        phase = (note_start - beat_times[index]) / interval
    else:
        interval = beat_times[index + 1] - beat_times[index]
        if interval <= 0:
            return None
        phase = (note_start - beat_times[index]) / interval

    return max(0.0, min(1.0, phase))


def build_melody_features(
    *,
    song_id: str,
    midi_path: Path,
    track_name: str,
    fallback_used: bool,
    notes: list[MelodyNote],
    beat_times: list[float],
    beat_source: Path | None,
) -> MelodyFeatures:
    """Build sequence features and summary statistics from extracted notes."""
    if not notes:
        raise ValueError(f"No melody notes found for song {song_id}.")

    sorted_notes = sorted(notes, key=lambda note: (note.start, note.end, note.pitch))
    pitches = [note.pitch for note in sorted_notes]
    durations = [note.duration for note in sorted_notes]
    intervals = [b - a for a, b in zip(pitches, pitches[1:])]
    contours = [contour_from_interval(interval) for interval in intervals]

    phases: list[float] = []
    for note in sorted_notes:
        phase = beat_phase(note.start, beat_times)
        if phase is not None:
            phases.append(phase)

    return MelodyFeatures(
        song_id=song_id,
        midi_path=midi_path,
        track_name=track_name,
        fallback_used=fallback_used,
        beat_source=beat_source,
        pitches=pitches,
        intervals=intervals,
        durations=durations,
        contours=contours,
        beat_positions=phases,
        note_count=len(sorted_notes),
        average_pitch=mean(pitches),
        average_duration=mean(durations),
        pitch_range=max(pitches) - min(pitches),
    )

