"""Streamlit demo UI for MelodyMatch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from melodymatch.config import DEFAULT_SONG_IDS, default_dataset_root
from melodymatch.pipeline import (
    compare_selected_pair,
    feature_summary_frame,
    load_selected_features,
    parse_song_ids,
)
from melodymatch.similarity import compare_all_pairs

XMIDI_CORE_FEATURES = [
    "note_count",
    "total_duration",
    "note_density",
    "average_note_duration",
    "duration_variance",
    "average_pitch",
    "pitch_range",
    "average_velocity",
    "average_polyphony",
    "max_polyphony",
    "tempo",
    "rhythmic_variability",
]


@st.cache_data(show_spinner=False)
def cached_pop909_features(dataset_root: str, song_ids_text: str):
    song_ids = parse_song_ids(song_ids_text)
    return load_selected_features(dataset_root, song_ids)


@st.cache_data(show_spinner=False)
def cached_xmidi_scan(dataset_root: str):
    from melodymatch.xmidi.dataset import scan_xmidi_dataset

    return scan_xmidi_dataset(dataset_root)


def show_similarity_metric_grid(result: dict[str, object]) -> None:
    """Display the selected POP909 pair's similarity scores."""
    st.metric("Final Similarity Score", f"{float(result['final_score']):.3f}")
    st.progress(float(result["final_score"]))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Interval", f"{float(result['interval_similarity']):.3f}")
    col2.metric("Duration", f"{float(result['duration_similarity']):.3f}")
    col3.metric("Contour", f"{float(result['contour_similarity']):.3f}")
    col4.metric("Beat", f"{float(result['beat_similarity']):.3f}")


def render_pop909_similarity_page() -> None:
    """Render the original POP909 melody similarity workflow."""
    st.header("POP909 Melody Similarity")

    with st.sidebar:
        st.subheader("POP909 Settings")
        dataset_root = st.text_input("POP909 dataset root", default_dataset_root())
        songs_text = st.text_input(
            "Selected songs",
            ",".join(DEFAULT_SONG_IDS),
            help="Comma-separated POP909 song ids.",
        )
        output_dir = st.text_input("Similarity output directory", "outputs")

    try:
        features = cached_pop909_features(dataset_root, songs_text)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    song_ids = sorted(features.keys())
    pairwise = compare_all_pairs(features)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    similarity_csv = output_path / "pairwise_similarity.csv"
    pairwise.to_csv(similarity_csv, index=False)

    st.subheader("Loaded Melody Tracks")
    st.dataframe(feature_summary_frame(features), use_container_width=True)

    st.subheader("Compare Two Songs")
    left, right = st.columns(2)
    song_a = left.selectbox("Song A", song_ids, index=0)
    default_b_index = 1 if len(song_ids) > 1 else 0
    song_b = right.selectbox("Song B", song_ids, index=default_b_index)

    if song_a == song_b:
        st.warning("Choose two different songs.")
        st.stop()

    result = compare_selected_pair(features, song_a, song_b)
    show_similarity_metric_grid(result)

    st.subheader("Pairwise Results")
    st.dataframe(pairwise, use_container_width=True)

    st.caption(f"Pairwise similarity CSV written to {similarity_csv}.")


def render_xmidi_genre_page() -> None:
    """Render the XMIDI symbolic MIDI genre-classification workflow."""
    st.header("XMIDI Genre Classification")

    with st.sidebar:
        st.subheader("XMIDI Settings")
        dataset_root = st.text_input("XMIDI dataset root", "XMIDI_Dataset")
        output_dir = st.text_input("Genre output directory", "outputs")
        max_depth = st.slider("Decision tree max depth", 2, 20, 8)
        test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
        limit_training = st.checkbox("Limit training files for demo speed", True)
        max_files = st.number_input(
            "Training file limit",
            min_value=50,
            value=1000,
            step=50,
            disabled=not limit_training,
        )
        report_baseline = st.checkbox("Report random forest baseline", True)

    output_path = Path(output_dir).expanduser().resolve()
    model_path = output_path / "xmidi_genre_decision_tree.joblib"

    try:
        metadata, warnings = cached_xmidi_scan(dataset_root)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if metadata.empty:
        st.warning("No valid XMIDI files were found.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("XMIDI Files", f"{len(metadata):,}")
    col2.metric("Genres", metadata["genre"].nunique())
    col3.metric("Emotions", metadata["emotion"].nunique())
    if warnings:
        st.caption(f"Skipped {len(warnings)} files with invalid XMIDI filenames.")

    st.subheader("Train Genre Model")
    if st.button("Train decision tree genre model"):
        from melodymatch.xmidi.pipeline import run_xmidi_training_pipeline

        progress_bar = st.progress(0.0)
        status = st.empty()

        def show_progress(index: int, total: int, filename: str) -> None:
            if index == 1 or index == total or index % 25 == 0:
                progress_bar.progress(index / total)
                status.text(f"Extracting features {index}/{total}: {filename}")

        try:
            with st.spinner("Training XMIDI genre classifier..."):
                result = run_xmidi_training_pipeline(
                    dataset_root=dataset_root,
                    output_dir=output_dir,
                    max_files=int(max_files) if limit_training else None,
                    reuse_features=False,
                    test_size=float(test_size),
                    max_depth=int(max_depth),
                    report_baseline=report_baseline,
                    progress_callback=show_progress,
                )
        except Exception as exc:
            st.error(str(exc))
        else:
            progress_bar.progress(1.0)
            status.text("Training complete.")
            st.success(f"Saved decision tree model to {result.model_path}")
            st.text(result.evaluation_text)

    st.subheader("Predict One XMIDI File")
    search = st.text_input("Filter filenames", "")
    choices = metadata
    if search:
        choices = choices[
            choices["filename"].str.contains(search, case=False, na=False)
            | choices["genre"].str.contains(search, case=False, na=False)
            | choices["emotion"].str.contains(search, case=False, na=False)
        ]

    if choices.empty:
        st.warning("No XMIDI files match that filter.")
        st.stop()

    limited_choices = choices.head(500).reset_index(drop=True)
    selected_filename = st.selectbox(
        "XMIDI file",
        limited_choices["filename"].tolist(),
    )
    selected_row = limited_choices.loc[limited_choices["filename"] == selected_filename].iloc[0]

    if not model_path.exists():
        st.info(f"Train a model first, or place a saved model at {model_path}.")
        st.stop()

    if st.button("Predict genre for selected file"):
        from melodymatch.xmidi.pipeline import predict_xmidi_file

        try:
            prediction = predict_xmidi_file(
                selected_row["midi_path"],
                model_path=model_path,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        pred_col, emotion_col, confidence_col = st.columns(3)
        pred_col.metric("Predicted Genre", str(prediction["predicted_genre"]))
        emotion_col.metric("Parsed Emotion", str(prediction["emotion"]))
        confidence = prediction.get("confidence")
        confidence_col.metric(
            "Model Confidence",
            "N/A" if confidence is None else f"{float(confidence):.3f}",
        )

        st.subheader("Core Extracted Features")
        feature_view = {
            column: prediction.get(column, 0.0) for column in XMIDI_CORE_FEATURES
        }
        st.dataframe(pd.DataFrame([feature_view]), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="MelodyMatch", layout="wide")
    st.title("MelodyMatch")

    with st.sidebar:
        page = st.radio(
            "Page",
            ["POP909 Melody Similarity", "XMIDI Genre Classification"],
        )

    if page == "POP909 Melody Similarity":
        render_pop909_similarity_page()
    else:
        render_xmidi_genre_page()


if __name__ == "__main__":
    main()

