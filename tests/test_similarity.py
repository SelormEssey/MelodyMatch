from dataclasses import replace
from pathlib import Path

import pytest

from melodymatch.models import MelodyFeatures
from melodymatch.similarity import compare_melodies, normalize_weights


@pytest.fixture
def melody() -> MelodyFeatures:
    return MelodyFeatures(
        song_id="sample",
        midi_path=Path("sample.mid"),
        track_name="MELODY",
        fallback_used=False,
        beat_source=None,
        pitches=[60, 62, 64, 62],
        intervals=[2, 2, -2],
        durations=[0.5, 0.5, 1.0, 0.5],
        contours=[1, 1, -1],
        beat_positions=[0.0, 0.5, 0.0, 0.5],
        note_count=4,
        average_pitch=62.0,
        average_duration=0.625,
        pitch_range=4,
    )


def test_identical_melodies_receive_perfect_score(melody: MelodyFeatures) -> None:
    result = compare_melodies(melody, melody)
    assert result["final_score"] == 1.0
    assert result["beat_available"] is True


def test_missing_beats_are_excluded_from_score(melody: MelodyFeatures) -> None:
    without_beats = replace(melody, beat_positions=[])
    result = compare_melodies(without_beats, without_beats)
    assert result["final_score"] == 1.0
    assert result["beat_similarity"] == 0.0
    assert result["beat_available"] is False


def test_weights_must_include_a_positive_value() -> None:
    with pytest.raises(ValueError, match="positive"):
        normalize_weights({"interval": 0.0, "duration": -1.0})
