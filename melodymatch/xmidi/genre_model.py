"""Decision-tree genre classifier for extracted XMIDI features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from melodymatch.xmidi.feature_extraction import FEATURE_COLUMNS


@dataclass
class GenreTrainingResult:
    """Outputs produced by XMIDI genre-classifier training."""

    model: DecisionTreeClassifier
    feature_table: pd.DataFrame
    predictions: pd.DataFrame
    evaluation_text: str
    tree_text: str
    feature_csv: Path
    predictions_csv: Path
    model_path: Path
    confusion_matrix_csv: Path


def _feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return numeric feature columns with invalid values replaced by 0."""
    return (
        data.loc[:, FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )


def _stratify_target(y: pd.Series, test_size: float) -> pd.Series | None:
    """Return y for stratification only when the split can support it."""
    counts = y.value_counts()
    class_count = len(counts)
    test_count = int(np.ceil(len(y) * test_size))
    train_count = len(y) - test_count
    if class_count > 1 and counts.min() >= 2 and test_count >= class_count:
        if train_count >= class_count:
            return y
    return None


def train_decision_tree_genre_classifier(
    feature_table: pd.DataFrame,
    *,
    output_dir: str | Path = "outputs",
    test_size: float = 0.2,
    max_depth: int = 8,
    random_state: int = 42,
    report_baseline: bool = True,
) -> GenreTrainingResult:
    """Train and evaluate a decision tree genre classifier."""
    required = {"filename", "emotion", "genre", *FEATURE_COLUMNS}
    missing = required.difference(feature_table.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Feature table is missing required columns: {missing_text}")

    data = feature_table.dropna(subset=["genre"]).reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("Need at least two valid XMIDI files to train a classifier.")
    if data["genre"].nunique() < 2:
        raise ValueError("Need at least two genres to train a classifier.")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    stratify = _stratify_target(data["genre"], test_size)
    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(_feature_frame(train_df), train_df["genre"])

    predicted = model.predict(_feature_frame(test_df))
    probabilities = (
        model.predict_proba(_feature_frame(test_df))
        if hasattr(model, "predict_proba")
        else None
    )
    confidence = probabilities.max(axis=1) if probabilities is not None else None

    predictions = test_df.loc[:, ["filename", "emotion", "genre", "file_id"]].copy()
    if "midi_path" in test_df.columns:
        predictions["midi_path"] = test_df["midi_path"].values
    predictions["predicted_genre"] = predicted
    predictions["confidence"] = confidence if confidence is not None else np.nan
    predictions["correct"] = predictions["genre"] == predictions["predicted_genre"]

    labels = sorted(data["genre"].unique())
    accuracy = accuracy_score(test_df["genre"], predicted)
    report = classification_report(test_df["genre"], predicted, zero_division=0)
    matrix = confusion_matrix(test_df["genre"], predicted, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)

    baseline_text = ""
    if report_baseline:
        baseline_text = _baseline_report(train_df, test_df, random_state=random_state)

    tree_text = export_text(model, feature_names=FEATURE_COLUMNS)
    evaluation_parts = [
        f"Decision tree accuracy: {accuracy:.4f}",
        "",
        "Classification report:",
        report,
        "Confusion matrix labels:",
        ", ".join(labels),
        str(matrix),
    ]
    if baseline_text:
        evaluation_parts.extend(["", baseline_text])
    evaluation_text = "\n".join(evaluation_parts)

    feature_csv = output_path / "xmidi_features.csv"
    predictions_csv = output_path / "xmidi_test_predictions.csv"
    model_path = output_path / "xmidi_genre_decision_tree.joblib"
    confusion_matrix_csv = output_path / "xmidi_confusion_matrix.csv"

    feature_table.to_csv(feature_csv, index=False)
    predictions.to_csv(predictions_csv, index=False)
    matrix_df.to_csv(confusion_matrix_csv)
    save_model(model, model_path)

    return GenreTrainingResult(
        model=model,
        feature_table=feature_table,
        predictions=predictions,
        evaluation_text=evaluation_text,
        tree_text=tree_text,
        feature_csv=feature_csv,
        predictions_csv=predictions_csv,
        model_path=model_path,
        confusion_matrix_csv=confusion_matrix_csv,
    )


def _baseline_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    random_state: int,
) -> str:
    """
    Train a small random forest as a simple baseline comparison.

    The decision tree remains the main model because it is easier to explain.
    """
    try:
        baseline = RandomForestClassifier(
            n_estimators=60,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        baseline.fit(_feature_frame(train_df), train_df["genre"])
        baseline_predicted = baseline.predict(_feature_frame(test_df))
        baseline_accuracy = accuracy_score(test_df["genre"], baseline_predicted)
        return f"Random forest baseline accuracy: {baseline_accuracy:.4f}"
    except Exception as exc:
        return f"Random forest baseline skipped: {exc}"


def save_model(model: DecisionTreeClassifier, model_path: str | Path) -> Path:
    """Save the trained decision tree with its feature-column contract."""
    path = Path(model_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, path)
    return path


def load_model(model_path: str | Path) -> dict[str, Any]:
    """Load a saved XMIDI genre model artifact."""
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Saved genre model not found: {path}")
    bundle = joblib.load(path)
    if "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError(f"Invalid model artifact: {path}")
    return bundle


def predict_genre(
    feature_row: dict[str, float],
    model_bundle: dict[str, Any],
) -> tuple[str, float | None]:
    """Predict a genre from one extracted feature row."""
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    frame = pd.DataFrame([{column: feature_row.get(column, 0.0) for column in feature_columns}])
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    prediction = str(model.predict(frame)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        confidence = float(np.max(probabilities))
    return prediction, confidence

