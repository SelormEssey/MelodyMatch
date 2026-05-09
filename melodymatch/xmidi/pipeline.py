"""High-level XMIDI feature, training, and prediction workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from melodymatch.xmidi.dataset import scan_xmidi_dataset
from melodymatch.xmidi.feature_extraction import FEATURE_COLUMNS, extract_xmidi_features
from melodymatch.xmidi.genre_model import (
    GenreTrainingResult,
    load_model,
    predict_genre,
    train_decision_tree_genre_classifier,
)
from melodymatch.xmidi.labels import parse_xmidi_filename

ProgressCallback = Callable[[int, int, str], None]


def build_xmidi_feature_table(
    *,
    dataset_root: str | Path = "XMIDI_Dataset",
    output_dir: str | Path = "outputs",
    max_files: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, Path, Path, list[str]]:
    """
    Scan XMIDI, extract symbolic MIDI features, and save the feature CSV.

    Invalid or messy MIDI files are skipped and recorded in an error CSV.
    """
    metadata, scan_warnings = scan_xmidi_dataset(dataset_root, max_files=max_files)
    if metadata.empty:
        raise ValueError(f"No valid XMIDI MIDI files found in {dataset_root}.")

    rows: list[dict[str, object]] = []
    errors: list[str] = list(scan_warnings)
    total = len(metadata)

    for index, item in enumerate(metadata.itertuples(index=False), start=1):
        if progress_callback:
            progress_callback(index, total, item.filename)
        try:
            features = extract_xmidi_features(item.midi_path)
        except Exception as exc:
            errors.append(f"{item.filename}: {exc}")
            continue

        rows.append(
            {
                "filename": item.filename,
                "emotion": item.emotion,
                "genre": item.genre,
                "file_id": item.file_id,
                "midi_path": item.midi_path,
                **features,
            }
        )

    if not rows:
        raise ValueError("No XMIDI files could be converted into feature rows.")

    feature_table = pd.DataFrame(rows)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    feature_csv = output_path / "xmidi_features.csv"
    error_csv = output_path / "xmidi_feature_errors.csv"
    feature_table.to_csv(feature_csv, index=False)
    pd.DataFrame({"error": errors}).to_csv(error_csv, index=False)

    return feature_table, feature_csv, error_csv, errors


def load_or_build_xmidi_feature_table(
    *,
    dataset_root: str | Path = "XMIDI_Dataset",
    output_dir: str | Path = "outputs",
    max_files: int | None = None,
    reuse_features: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, Path, Path, list[str]]:
    """Reuse the saved XMIDI feature CSV when requested; otherwise rebuild it."""
    output_path = Path(output_dir).expanduser().resolve()
    feature_csv = output_path / "xmidi_features.csv"
    error_csv = output_path / "xmidi_feature_errors.csv"

    if reuse_features and feature_csv.exists():
        return pd.read_csv(feature_csv), feature_csv, error_csv, []

    return build_xmidi_feature_table(
        dataset_root=dataset_root,
        output_dir=output_dir,
        max_files=max_files,
        progress_callback=progress_callback,
    )


def run_xmidi_training_pipeline(
    *,
    dataset_root: str | Path = "XMIDI_Dataset",
    output_dir: str | Path = "outputs",
    max_files: int | None = None,
    reuse_features: bool = False,
    test_size: float = 0.2,
    max_depth: int = 8,
    random_state: int = 42,
    report_baseline: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> GenreTrainingResult:
    """Build features, train the decision tree, save outputs, and return results."""
    feature_table, _feature_csv, _error_csv, _errors = load_or_build_xmidi_feature_table(
        dataset_root=dataset_root,
        output_dir=output_dir,
        max_files=max_files,
        reuse_features=reuse_features,
        progress_callback=progress_callback,
    )

    return train_decision_tree_genre_classifier(
        feature_table,
        output_dir=output_dir,
        test_size=test_size,
        max_depth=max_depth,
        random_state=random_state,
        report_baseline=report_baseline,
    )


def predict_xmidi_file(
    midi_path: str | Path,
    *,
    model_path: str | Path = "outputs/xmidi_genre_decision_tree.joblib",
) -> dict[str, object]:
    """Extract features for one XMIDI file and predict its genre."""
    path = Path(midi_path).expanduser().resolve()
    labels = parse_xmidi_filename(path)
    features = extract_xmidi_features(path)
    model_bundle = load_model(model_path)
    predicted_genre, confidence = predict_genre(features, model_bundle)

    return {
        "filename": labels.filename,
        "emotion": labels.emotion,
        "true_genre": labels.genre,
        "file_id": labels.file_id,
        "midi_path": str(path),
        "predicted_genre": predicted_genre,
        "confidence": confidence,
        **{column: features.get(column, 0.0) for column in FEATURE_COLUMNS},
    }

