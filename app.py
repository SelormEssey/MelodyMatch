"""Interactive Streamlit interface for MelodyMatch."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from melodymatch.config import DEFAULT_SONG_IDS, default_dataset_root
from melodymatch.midi_loader import load_uploaded_midi
from melodymatch.models import MelodyFeatures
from melodymatch.pipeline import feature_summary_frame, load_selected_features
from melodymatch.similarity import compare_all_pairs, compare_melodies


@st.cache_data(show_spinner=False)
def cached_demo_features() -> dict[str, MelodyFeatures]:
    """Load the demonstration melodies bundled with the repository."""
    return load_selected_features(default_dataset_root(), DEFAULT_SONG_IDS)


@st.cache_data(show_spinner=False)
def cached_upload(content: bytes, filename: str) -> MelodyFeatures:
    """Extract a melody from one uploaded file and cache the result."""
    return load_uploaded_midi(content, filename)


def similarity_label(score: float) -> str:
    """Translate a numeric score into a readable comparison label."""
    if score >= 0.85:
        return "Very similar"
    if score >= 0.70:
        return "Moderately similar"
    if score >= 0.50:
        return "Some shared structure"
    return "Structurally different"


def component_scores(result: dict[str, object]) -> dict[str, float]:
    """Return available component scores with reader-friendly names."""
    scores = {
        "Pitch intervals": float(result["interval_similarity"]),
        "Note duration": float(result["duration_similarity"]),
        "Melodic contour": float(result["contour_similarity"]),
    }
    if bool(result["beat_available"]):
        scores["Beat placement"] = float(result["beat_similarity"])
    return scores


def show_score_summary(result: dict[str, object]) -> None:
    """Display the overall result and explain its strongest signals."""
    score = float(result["final_score"])
    scores = component_scores(result)
    strongest = max(scores, key=scores.get)
    weakest = min(scores, key=scores.get)

    score_col, summary_col = st.columns([1, 2])
    with score_col:
        st.metric("Overall similarity", f"{score:.1%}")
        st.progress(score)
        st.caption(similarity_label(score))
    with summary_col:
        st.markdown("#### What drove this result")
        st.write(
            f"**{strongest}** is the strongest match at {scores[strongest]:.1%}. "
            f"**{weakest}** differs the most at {scores[weakest]:.1%}."
        )
        if not bool(result["beat_available"]):
            st.info(
                "Beat annotations were unavailable, so the overall score was "
                "reweighted across pitch intervals, duration, and contour."
            )

    columns = st.columns(len(scores))
    for column, (name, value) in zip(columns, scores.items()):
        column.metric(name, f"{value:.1%}")


def pitch_frame(left: MelodyFeatures, right: MelodyFeatures) -> pd.DataFrame:
    """Build a chart-ready frame of pitch sequences."""
    length = max(len(left.pitches), len(right.pitches))
    left_name = f"{left.song_id} A" if left.song_id == right.song_id else left.song_id
    right_name = f"{right.song_id} B" if left.song_id == right.song_id else right.song_id
    return pd.DataFrame(
        {
            left_name: pd.Series(left.pitches, index=range(len(left.pitches))),
            right_name: pd.Series(right.pitches, index=range(len(right.pitches))),
        },
        index=range(length),
    )


def duration_frame(left: MelodyFeatures, right: MelodyFeatures) -> pd.DataFrame:
    """Build a chart-ready frame of the first 80 note durations."""
    visible_notes = 80
    length = min(max(len(left.durations), len(right.durations)), visible_notes)
    left_name = f"{left.song_id} A" if left.song_id == right.song_id else left.song_id
    right_name = f"{right.song_id} B" if left.song_id == right.song_id else right.song_id
    return pd.DataFrame(
        {
            left_name: pd.Series(left.durations[:length]),
            right_name: pd.Series(right.durations[:length]),
        }
    )


def show_visual_comparison(left: MelodyFeatures, right: MelodyFeatures) -> None:
    """Render pitch-shape and rhythm charts for the selected melodies."""
    st.subheader("See the melodies")
    pitch_tab, rhythm_tab = st.tabs(["Pitch shape", "Note duration"])

    with pitch_tab:
        st.line_chart(pitch_frame(left, right), height=320)
        st.caption(
            "Each line follows MIDI pitch over the note sequence. Similar shapes "
            "suggest related melodic movement even when the melodies use different keys."
        )
    with rhythm_tab:
        st.line_chart(duration_frame(left, right), height=320)
        st.caption("The first 80 note lengths are shown in seconds.")


def show_pair_details(
    left: MelodyFeatures,
    right: MelodyFeatures,
    result: dict[str, object],
) -> None:
    """Display summary differences and extraction transparency."""
    with st.expander("Technical details"):
        differences = pd.DataFrame(
            {
                "Measure": [
                    "Note-count difference",
                    "Average-pitch difference",
                    "Average-duration difference",
                    "Pitch-range difference",
                ],
                "Difference": [
                    int(result["note_count_difference"]),
                    round(float(result["average_pitch_difference"]), 2),
                    round(float(result["average_duration_difference"]), 3),
                    int(result["pitch_range_difference"]),
                ],
            }
        )
        st.dataframe(differences, hide_index=True, width="stretch")

        for melody in (left, right):
            if melody.fallback_used:
                st.caption(
                    f"{melody.song_id}: no track was labeled MELODY, so MelodyMatch "
                    "selected the highest-pitched non-drum track with notes."
                )


def choose_demo_pair() -> tuple[MelodyFeatures, MelodyFeatures, dict[str, MelodyFeatures]]:
    """Let the user select two bundled POP909 examples."""
    features = cached_demo_features()
    song_ids = sorted(features)
    left_col, right_col = st.columns(2)
    song_a = left_col.selectbox("First melody", song_ids, index=0)
    song_b = right_col.selectbox("Second melody", song_ids, index=1)
    if song_a == song_b:
        st.warning("Choose two different melodies to compare.")
        st.stop()
    return features[song_a], features[song_b], features


def choose_uploaded_pair() -> tuple[MelodyFeatures, MelodyFeatures]:
    """Let the user upload two MIDI files for comparison."""
    left_col, right_col = st.columns(2)
    first = left_col.file_uploader("First MIDI", type=["mid", "midi"], key="first")
    second = right_col.file_uploader("Second MIDI", type=["mid", "midi"], key="second")
    if first is None or second is None:
        st.info("Upload two MIDI files to begin the comparison.")
        st.stop()

    try:
        left = cached_upload(first.getvalue(), first.name)
        right = cached_upload(second.getvalue(), second.name)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    return left, right


def main() -> None:
    st.set_page_config(page_title="MelodyMatch", page_icon="🎵", layout="wide")
    st.title("🎵 MelodyMatch")
    st.write(
        "Compare two MIDI melodies and explore how pitch, rhythm, contour, and "
        "beat placement shape their similarity."
    )

    mode = st.radio(
        "Choose a comparison mode",
        ["Try the demo", "Upload MIDI files"],
        horizontal=True,
    )
    st.divider()

    if mode == "Try the demo":
        left, right, demos = choose_demo_pair()
    else:
        left, right = choose_uploaded_pair()
        demos = None

    result = compare_melodies(left, right)
    show_score_summary(result)
    show_visual_comparison(left, right)
    show_pair_details(left, right, result)

    if demos is not None:
        with st.expander("Explore all demonstration comparisons"):
            st.dataframe(compare_all_pairs(demos), hide_index=True, width="stretch")
            st.dataframe(feature_summary_frame(demos), hide_index=True, width="stretch")

    st.divider()
    st.caption(
        "MelodyMatch analyzes symbolic MIDI note events using dynamic time warping. "
        "Demonstration melodies are licensed excerpts from POP909."
    )


if __name__ == "__main__":
    main()
