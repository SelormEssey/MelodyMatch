"""Feature-based melody similarity scoring."""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Iterable

import pandas as pd

from melodymatch.config import DEFAULT_SIMILARITY_WEIGHTS
from melodymatch.models import MelodyFeatures

DistanceFunction = Callable[[float, float], float]


def normalize_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    """Normalize similarity weights while preserving their names."""
    selected = dict(DEFAULT_SIMILARITY_WEIGHTS if weights is None else weights)
    total = sum(max(0.0, value) for value in selected.values())
    if total <= 0:
        raise ValueError("At least one similarity weight must be positive.")
    return {name: max(0.0, value) / total for name, value in selected.items()}


def _as_float_list(sequence: Iterable[float]) -> list[float]:
    return [float(value) for value in sequence]


def dtw_average_distance(
    sequence_a: Iterable[float],
    sequence_b: Iterable[float],
    distance: DistanceFunction,
) -> float | None:
    """
    Return dynamic-time-warping distance normalized by the longer sequence.

    None means the comparison is unavailable because exactly one sequence is
    empty. Two empty sequences are treated as a perfect match.
    """
    a = _as_float_list(sequence_a)
    b = _as_float_list(sequence_b)

    if not a and not b:
        return 0.0
    if not a or not b:
        return None

    previous = [float("inf")] * (len(b) + 1)
    previous[0] = 0.0

    for item_a in a:
        current = [float("inf")] * (len(b) + 1)
        for index_b, item_b in enumerate(b, start=1):
            cost = distance(item_a, item_b)
            current[index_b] = cost + min(
                previous[index_b],
                current[index_b - 1],
                previous[index_b - 1],
            )
        previous = current

    return previous[-1] / max(len(a), len(b))


def scaled_similarity(
    sequence_a: Iterable[float],
    sequence_b: Iterable[float],
    *,
    distance: DistanceFunction,
    scale: float,
    unavailable_score: float = 0.0,
) -> float:
    """Convert normalized DTW distance to a 0-1 similarity score."""
    average_distance = dtw_average_distance(sequence_a, sequence_b, distance)
    if average_distance is None:
        return unavailable_score
    if scale <= 0:
        raise ValueError("Similarity scale must be positive.")
    return max(0.0, min(1.0, 1.0 - (average_distance / scale)))


def normalized_duration_sequence(durations: list[float]) -> list[float]:
    """Normalize durations by their mean so tempo scale matters less."""
    if not durations:
        return []
    average = sum(durations) / len(durations)
    if average <= 0:
        return []
    return [duration / average for duration in durations]


def circular_phase_distance(a: float, b: float) -> float:
    """Distance between two beat phases on a circular 0-1 scale."""
    direct = abs(a - b)
    return min(direct, 1.0 - direct)


def compare_melodies(
    melody_a: MelodyFeatures,
    melody_b: MelodyFeatures,
    weights: dict[str, float] | None = None,
) -> dict[str, float | int | str]:
    """Compare two extracted melodies and return pairwise features."""
    active_weights = normalize_weights(weights)

    interval_similarity = scaled_similarity(
        melody_a.intervals,
        melody_b.intervals,
        distance=lambda a, b: min(abs(a - b), 12.0),
        scale=12.0,
    )
    duration_similarity = scaled_similarity(
        normalized_duration_sequence(melody_a.durations),
        normalized_duration_sequence(melody_b.durations),
        distance=lambda a, b: min(abs(a - b), 2.0),
        scale=2.0,
    )
    contour_similarity = scaled_similarity(
        melody_a.contours,
        melody_b.contours,
        distance=lambda a, b: 0.0 if int(a) == int(b) else 1.0,
        scale=1.0,
    )
    beat_similarity = scaled_similarity(
        melody_a.beat_positions,
        melody_b.beat_positions,
        distance=circular_phase_distance,
        scale=0.5,
        unavailable_score=0.5,
    )

    final_score = (
        active_weights.get("interval", 0.0) * interval_similarity
        + active_weights.get("duration", 0.0) * duration_similarity
        + active_weights.get("contour", 0.0) * contour_similarity
        + active_weights.get("beat", 0.0) * beat_similarity
    )

    return {
        "song_a": melody_a.song_id,
        "song_b": melody_b.song_id,
        "interval_similarity": round(interval_similarity, 6),
        "duration_similarity": round(duration_similarity, 6),
        "contour_similarity": round(contour_similarity, 6),
        "beat_similarity": round(beat_similarity, 6),
        "final_score": round(final_score, 6),
        "note_count_difference": abs(melody_a.note_count - melody_b.note_count),
        "average_pitch_difference": round(
            abs(melody_a.average_pitch - melody_b.average_pitch), 6
        ),
        "average_duration_difference": round(
            abs(melody_a.average_duration - melody_b.average_duration), 6
        ),
        "pitch_range_difference": abs(melody_a.pitch_range - melody_b.pitch_range),
    }


def compare_all_pairs(
    melodies: dict[str, MelodyFeatures],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compare every unique pair of selected songs."""
    rows = [
        compare_melodies(melodies[a], melodies[b], weights=weights)
        for a, b in combinations(sorted(melodies.keys()), 2)
    ]
    return pd.DataFrame(rows)

