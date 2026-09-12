"""Load POP909 MIDI files and extract the melody track."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import pretty_midi

from melodymatch.config import normalize_song_id
from melodymatch.features import build_melody_features
from melodymatch.models import MelodyFeatures, MelodyNote

FLOAT_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def find_midi_path(dataset_root: Path, song_id: str) -> Path:
    """Find the expected POP909 MIDI path for one song."""
    song_dir = dataset_root / song_id
    expected = song_dir / f"{song_id}.mid"
    if expected.exists():
        return expected

    candidates = sorted(song_dir.glob("*.mid"))
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find {expected}. Expected POP909 layout: "
        f"{song_id}/{song_id}.mid"
    )


def choose_melody_track(
    midi_data: pretty_midi.PrettyMIDI,
) -> tuple[pretty_midi.Instrument, bool]:
    """
    Choose the POP909 melody instrument.

    Main rule: use the instrument whose name contains MELODY.
    Fallback: use the non-drum instrument with the highest average pitch, with
    note count as a tie-breaker. This is documented in the README because it is
    a heuristic, not part of the dataset definition.
    """
    named_matches = [
        instrument
        for instrument in midi_data.instruments
        if "MELODY" in (instrument.name or "").upper() and instrument.notes
    ]
    if named_matches:
        return max(named_matches, key=lambda instrument: len(instrument.notes)), False

    candidates = [
        instrument
        for instrument in midi_data.instruments
        if not instrument.is_drum and instrument.notes
    ]
    if not candidates:
        raise ValueError("No non-drum MIDI tracks with notes were found.")

    def fallback_score(instrument: pretty_midi.Instrument) -> tuple[float, int]:
        average_pitch = sum(note.pitch for note in instrument.notes) / len(
            instrument.notes
        )
        return average_pitch, len(instrument.notes)

    return max(candidates, key=fallback_score), True


def parse_beat_file(path: Path, midi_data: pretty_midi.PrettyMIDI) -> list[float]:
    """
    Parse POP909 beat_midi.txt or beat_audio.txt.

    The first numeric value on each row is treated as the beat time. If the
    values look like MIDI ticks rather than seconds, they are converted to time
    using pretty_midi's tick_to_time method.
    """
    raw_values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = FLOAT_PATTERN.search(line)
        if match:
            raw_values.append(float(match.group(0)))

    if not raw_values:
        return []

    midi_end = midi_data.get_end_time()
    mostly_integer = sum(float(value).is_integer() for value in raw_values) / len(
        raw_values
    )
    looks_like_ticks = (
        max(raw_values) > max(10.0, midi_end * 2.5) and mostly_integer > 0.8
    )

    if looks_like_ticks:
        beat_times = [
            float(midi_data.tick_to_time(int(round(value)))) for value in raw_values
        ]
    else:
        beat_times = raw_values

    cleaned = sorted({round(value, 6) for value in beat_times if value >= 0})
    return cleaned


def find_beat_file(song_dir: Path) -> Path | None:
    """Prefer beat_midi.txt, then fall back to beat_audio.txt."""
    midi_beats = song_dir / "beat_midi.txt"
    if midi_beats.exists():
        return midi_beats

    audio_beats = song_dir / "beat_audio.txt"
    if audio_beats.exists():
        return audio_beats

    return None


def instrument_notes(instrument: pretty_midi.Instrument) -> list[MelodyNote]:
    """Convert pretty_midi notes into internal note records."""
    notes: list[MelodyNote] = []
    for note in instrument.notes:
        duration = float(note.end - note.start)
        if duration <= 0:
            continue
        notes.append(
            MelodyNote(
                pitch=int(note.pitch),
                start=float(note.start),
                end=float(note.end),
                duration=duration,
            )
        )
    return sorted(notes, key=lambda item: (item.start, item.end, item.pitch))


def load_song_features(dataset_root: str | Path, song_id: str | int) -> MelodyFeatures:
    """Load one POP909 song and return extracted melody features."""
    root = Path(dataset_root).expanduser().resolve()
    normalized_id = normalize_song_id(song_id)
    midi_path = find_midi_path(root, normalized_id)
    song_dir = midi_path.parent

    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        raise RuntimeError(f"Could not parse MIDI file {midi_path}: {exc}") from exc

    instrument, fallback_used = choose_melody_track(midi_data)
    beat_source = find_beat_file(song_dir)
    beat_times = parse_beat_file(beat_source, midi_data) if beat_source else []
    notes = instrument_notes(instrument)

    track_name = instrument.name.strip() if instrument.name else "Unnamed track"
    return build_melody_features(
        song_id=normalized_id,
        midi_path=midi_path,
        track_name=track_name,
        fallback_used=fallback_used,
        notes=notes,
        beat_times=beat_times,
        beat_source=beat_source,
    )


def load_uploaded_midi(content: bytes, filename: str) -> MelodyFeatures:
    """Extract melody features from an uploaded MIDI file."""
    if not content:
        raise ValueError("The uploaded MIDI file is empty.")

    safe_name = Path(filename).name
    try:
        midi_data = pretty_midi.PrettyMIDI(BytesIO(content))
    except Exception as exc:
        raise RuntimeError(f"Could not parse {safe_name} as MIDI: {exc}") from exc

    instrument, fallback_used = choose_melody_track(midi_data)
    notes = instrument_notes(instrument)
    track_name = instrument.name.strip() if instrument.name else "Detected melody track"

    return build_melody_features(
        song_id=Path(safe_name).stem,
        midi_path=Path(safe_name),
        track_name=track_name,
        fallback_used=fallback_used,
        notes=notes,
        beat_times=[],
        beat_source=None,
    )
