"""High-level pipeline functions used by the CLI and UI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from melodymatch.config import DEFAULT_SONG_IDS, normalize_song_id
from melodymatch.midi_loader import load_song_features
from melodymatch.models import MelodyFeatures
from melodymatch.similarity import compare_all_pairs, compare_melodies


def parse_song_ids(song_ids: list[str] | tuple[str, ...] | str | None) -> list[str]:
    """Parse song ids from a list or comma-separated string."""
    if song_ids is None:
        return list(DEFAULT_SONG_IDS)
    if isinstance(song_ids, str):
        pieces = [piece.strip() for piece in song_ids.replace(" ", ",").split(",")]
        return [normalize_song_id(piece) for piece in pieces if piece]
    return [normalize_song_id(song_id) for song_id in song_ids]


def load_selected_features(
    dataset_root: str | Path,
    song_ids: list[str] | tuple[str, ...] | str | None = None,
) -> dict[str, MelodyFeatures]:
    """Load feature objects for all selected songs."""
    selected_ids = parse_song_ids(song_ids)
    if len(selected_ids) < 2:
        raise ValueError("Select at least two songs for pairwise comparison.")

    features: dict[str, MelodyFeatures] = {}
    errors: list[str] = []

    for song_id in selected_ids:
        try:
            features[song_id] = load_song_features(dataset_root, song_id)
        except Exception as exc:
            errors.append(f"{song_id}: {exc}")

    if errors:
        details = "\n".join(errors)
        raise RuntimeError(f"Could not load all selected songs:\n{details}")

    return features


def feature_summary_frame(features: dict[str, MelodyFeatures]) -> pd.DataFrame:
    """Return a DataFrame describing loaded melody tracks."""
    return pd.DataFrame([feature.summary() for feature in features.values()])


def compare_selected_pair(
    features: dict[str, MelodyFeatures],
    song_a: str,
    song_b: str,
) -> dict[str, float | int | str]:
    """Compare two already-loaded songs."""
    normalized_a = normalize_song_id(song_a)
    normalized_b = normalize_song_id(song_b)
    if normalized_a == normalized_b:
        raise ValueError("Choose two different songs.")
    return compare_melodies(features[normalized_a], features[normalized_b])


def run_similarity_pipeline(
    *,
    dataset_root: str | Path,
    song_ids: list[str] | tuple[str, ...] | str | None = None,
    output_dir: str | Path = "outputs",
) -> tuple[dict[str, MelodyFeatures], pd.DataFrame, Path]:
    """Load songs, compare all pairs, and write the similarity CSV."""
    features = load_selected_features(dataset_root, song_ids)
    pairwise = compare_all_pairs(features)

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    similarity_csv = output_path / "pairwise_similarity.csv"
    pairwise.to_csv(similarity_csv, index=False)

    return features, pairwise, similarity_csv
