"""Filename label parsing for the XMIDI dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class XMIDILabels:
    """Labels encoded in an XMIDI filename."""

    filename: str
    emotion: str
    genre: str
    file_id: str


def parse_xmidi_filename(path: str | Path) -> XMIDILabels:
    """
    Parse filenames in the form XMIDI_<Emotion>_<Genre>_<ID>.midi.

    The genre label is intentionally read from the filename because the local
    XMIDI folder does not require a separate metadata file.
    """
    filename = Path(path).name
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 4 or parts[0] != "XMIDI":
        raise ValueError(
            f"Invalid XMIDI filename '{filename}'. Expected "
            "XMIDI_<Emotion>_<Genre>_<ID>.midi"
        )

    return XMIDILabels(
        filename=filename,
        emotion=parts[1].lower(),
        genre=parts[2].lower(),
        file_id="_".join(parts[3:]),
    )

