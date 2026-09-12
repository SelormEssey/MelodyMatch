from pathlib import Path

import pytest

from melodymatch.midi_loader import load_song_features, load_uploaded_midi


DATA_ROOT = Path("data/POP909")


def test_bundled_song_loads_named_melody_track() -> None:
    melody = load_song_features(DATA_ROOT, "001")
    assert melody.song_id == "001"
    assert melody.track_name == "MELODY"
    assert melody.fallback_used is False
    assert melody.note_count == 264
    assert melody.beat_positions


def test_uploaded_midi_loads_without_dataset_metadata() -> None:
    content = (DATA_ROOT / "002" / "002.mid").read_bytes()
    melody = load_uploaded_midi(content, "my-song.mid")
    assert melody.song_id == "my-song"
    assert melody.note_count == 310
    assert melody.beat_positions == []


def test_empty_upload_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        load_uploaded_midi(b"", "empty.mid")


def test_invalid_upload_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="Could not parse"):
        load_uploaded_midi(b"not midi data", "broken.mid")
