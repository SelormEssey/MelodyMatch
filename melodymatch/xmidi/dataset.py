"""Dataset scanning utilities for local XMIDI MIDI files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from melodymatch.xmidi.labels import parse_xmidi_filename

SUPPORTED_EXTENSIONS = {".mid", ".midi"}


def iter_xmidi_files(dataset_root: str | Path):
    """Yield supported MIDI files from an XMIDI dataset folder."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"XMIDI dataset folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"XMIDI dataset path is not a folder: {root}")

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def scan_xmidi_dataset(
    dataset_root: str | Path = "XMIDI_Dataset",
    *,
    max_files: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Scan XMIDI files and return parsed filename labels.

    Invalid filenames are skipped and returned as warning strings.
    """
    rows: list[dict[str, str]] = []
    warnings: list[str] = []

    for path in iter_xmidi_files(dataset_root):
        try:
            labels = parse_xmidi_filename(path)
        except ValueError as exc:
            warnings.append(str(exc))
            continue

        rows.append(
            {
                "filename": labels.filename,
                "emotion": labels.emotion,
                "genre": labels.genre,
                "file_id": labels.file_id,
                "midi_path": str(path),
            }
        )

    metadata = pd.DataFrame(rows)
    if max_files is not None and len(metadata) > max_files:
        metadata = balanced_metadata_sample(metadata, max_files=max_files)

    return metadata.reset_index(drop=True), warnings


def balanced_metadata_sample(metadata: pd.DataFrame, *, max_files: int) -> pd.DataFrame:
    """
    Pick a deterministic genre-aware sample from the metadata table.

    This avoids the problem where an alphabetical filename sample contains only
    one genre, which would make classifier training impossible.
    """
    if max_files <= 0 or metadata.empty:
        return metadata.head(0)

    genres = sorted(metadata["genre"].dropna().unique())
    if not genres:
        return metadata.head(max_files)

    per_genre = max(1, max_files // len(genres))
    sampled_parts = []

    for genre in genres:
        group = metadata[metadata["genre"] == genre]
        count = min(len(group), per_genre)
        if count:
            sampled_parts.append(group.sample(n=count, random_state=42))

    sampled = pd.concat(sampled_parts) if sampled_parts else metadata.head(0)
    remaining = max_files - len(sampled)
    if remaining > 0:
        leftovers = metadata.drop(sampled.index)
        if not leftovers.empty:
            sampled = pd.concat(
                [
                    sampled,
                    leftovers.sample(
                        n=min(remaining, len(leftovers)),
                        random_state=42,
                    ),
                ]
            )

    return sampled.sort_values(["genre", "filename"]).head(max_files)
