"""Symbolic MIDI feature extraction for XMIDI genre classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

INTERVAL_HISTOGRAM_COLUMNS = [
    *(f"interval_abs_{value}_ratio" for value in range(12)),
    "interval_abs_12plus_ratio",
]

FEATURE_COLUMNS = [
    "note_count",
    "non_drum_note_count",
    "drum_note_ratio",
    "total_duration",
    "note_density",
    "average_note_duration",
    "duration_variance",
    "average_pitch",
    "pitch_variance",
    "pitch_range",
    "average_velocity",
    "velocity_variance",
    "average_polyphony",
    "max_polyphony",
    "overlap_ratio",
    "tempo",
    "tempo_change_count",
    "mean_ioi",
    "ioi_variance",
    "rhythmic_variability",
    "interval_mean",
    "interval_std",
    "abs_interval_mean",
    "contour_up_ratio",
    "contour_down_ratio",
    "contour_same_ratio",
    *INTERVAL_HISTOGRAM_COLUMNS,
]


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_var(values: list[float]) -> float:
    return float(np.var(values)) if values else 0.0


def _collect_notes(midi_data: pretty_midi.PrettyMIDI):
    """Return all notes, non-drum notes, and drum-note count."""
    all_notes = []
    non_drum_notes = []
    drum_note_count = 0

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            if note.end <= note.start:
                continue
            all_notes.append(note)
            if instrument.is_drum:
                drum_note_count += 1
            else:
                non_drum_notes.append(note)

    all_notes.sort(key=lambda note: (note.start, note.end, note.pitch))
    non_drum_notes.sort(key=lambda note: (note.start, note.end, note.pitch))
    return all_notes, non_drum_notes, drum_note_count


def _polyphony_features(notes, total_duration: float) -> tuple[float, int, float]:
    """Estimate average/max simultaneous notes and note overlap ratio."""
    if not notes or total_duration <= 0:
        return 0.0, 0, 0.0

    events = []
    for note in notes:
        events.append((float(note.start), 1))
        events.append((float(note.end), -1))
    events.sort(key=lambda item: (item[0], item[1]))

    active = 0
    max_active = 0
    active_area = 0.0
    previous_time = events[0][0]
    for time, delta in events:
        if time > previous_time:
            active_area += active * (time - previous_time)
            previous_time = time
        active += delta
        max_active = max(max_active, active)

    average_polyphony = active_area / total_duration

    overlapping_notes = 0
    latest_end = 0.0
    for note in notes:
        if note.start < latest_end:
            overlapping_notes += 1
        latest_end = max(latest_end, float(note.end))

    overlap_ratio = overlapping_notes / len(notes)
    return float(average_polyphony), int(max_active), float(overlap_ratio)


def _interval_features(notes) -> dict[str, float]:
    """Compute interval, contour, and interval-histogram features."""
    pitch_notes = sorted(notes, key=lambda note: (note.start, note.end, note.pitch))
    pitches = [int(note.pitch) for note in pitch_notes]
    intervals = [b - a for a, b in zip(pitches, pitches[1:])]
    abs_intervals = [abs(value) for value in intervals]

    features = {
        "interval_mean": _safe_mean([float(value) for value in intervals]),
        "interval_std": float(np.std(intervals)) if intervals else 0.0,
        "abs_interval_mean": _safe_mean([float(value) for value in abs_intervals]),
        "contour_up_ratio": 0.0,
        "contour_down_ratio": 0.0,
        "contour_same_ratio": 0.0,
    }

    if intervals:
        total = len(intervals)
        features["contour_up_ratio"] = sum(value > 0 for value in intervals) / total
        features["contour_down_ratio"] = sum(value < 0 for value in intervals) / total
        features["contour_same_ratio"] = sum(value == 0 for value in intervals) / total

    histogram_counts = {column: 0 for column in INTERVAL_HISTOGRAM_COLUMNS}
    for interval in abs_intervals:
        column = (
            f"interval_abs_{interval}_ratio"
            if interval < 12
            else "interval_abs_12plus_ratio"
        )
        histogram_counts[column] += 1

    total_intervals = len(abs_intervals)
    for column in INTERVAL_HISTOGRAM_COLUMNS:
        features[column] = (
            histogram_counts[column] / total_intervals if total_intervals else 0.0
        )

    return features


def extract_xmidi_features(midi_path: str | Path) -> dict[str, float]:
    """
    Extract explainable symbolic features from one MIDI file.

    Raises ValueError when the MIDI has no usable note events.
    """
    path = Path(midi_path).expanduser().resolve()
    try:
        midi_data = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        raise ValueError(f"Could not parse MIDI file {path.name}: {exc}") from exc

    all_notes, non_drum_notes, drum_note_count = _collect_notes(midi_data)
    if not all_notes:
        raise ValueError(f"No valid note events found in {path.name}.")

    pitch_notes = non_drum_notes or all_notes
    total_duration = float(midi_data.get_end_time())
    if total_duration <= 0:
        total_duration = max(float(note.end) for note in all_notes)

    durations = [float(note.end - note.start) for note in all_notes]
    pitches = [float(note.pitch) for note in pitch_notes]
    velocities = [float(note.velocity) for note in all_notes]
    starts = sorted({round(float(note.start), 6) for note in all_notes})
    iois = [b - a for a, b in zip(starts, starts[1:]) if b > a]
    mean_ioi = _safe_mean(iois)
    ioi_variance = _safe_var(iois)
    rhythmic_variability = (
        float(np.std(iois) / mean_ioi) if iois and mean_ioi > 0 else 0.0
    )

    average_polyphony, max_polyphony, overlap_ratio = _polyphony_features(
        all_notes,
        total_duration,
    )

    try:
        tempo = float(midi_data.estimate_tempo())
    except Exception:
        tempo = 0.0
    if not np.isfinite(tempo):
        tempo = 0.0

    try:
        tempo_times, _tempo_values = midi_data.get_tempo_changes()
        tempo_change_count = max(0, len(tempo_times) - 1)
    except Exception:
        tempo_change_count = 0

    features = {
        "note_count": float(len(all_notes)),
        "non_drum_note_count": float(len(non_drum_notes)),
        "drum_note_ratio": float(drum_note_count / len(all_notes)),
        "total_duration": total_duration,
        "note_density": float(len(all_notes) / total_duration)
        if total_duration > 0
        else 0.0,
        "average_note_duration": _safe_mean(durations),
        "duration_variance": _safe_var(durations),
        "average_pitch": _safe_mean(pitches),
        "pitch_variance": _safe_var(pitches),
        "pitch_range": float(max(pitches) - min(pitches)) if pitches else 0.0,
        "average_velocity": _safe_mean(velocities),
        "velocity_variance": _safe_var(velocities),
        "average_polyphony": average_polyphony,
        "max_polyphony": float(max_polyphony),
        "overlap_ratio": overlap_ratio,
        "tempo": tempo,
        "tempo_change_count": float(tempo_change_count),
        "mean_ioi": mean_ioi,
        "ioi_variance": ioi_variance,
        "rhythmic_variability": rhythmic_variability,
    }
    features.update(_interval_features(pitch_notes))

    return {column: float(features.get(column, 0.0)) for column in FEATURE_COLUMNS}

