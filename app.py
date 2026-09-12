"""Streamlit interface for MelodyMatch."""

from __future__ import annotations

import streamlit as st

from melodymatch.config import DEFAULT_SONG_IDS, default_dataset_root
from melodymatch.pipeline import compare_selected_pair, feature_summary_frame, load_selected_features
from melodymatch.similarity import compare_all_pairs


@st.cache_data(show_spinner=False)
def cached_features():
    """Load the demonstration melodies bundled with the repository."""
    return load_selected_features(default_dataset_root(), DEFAULT_SONG_IDS)


def show_similarity_metric_grid(result: dict[str, object]) -> None:
    """Display the selected pair's overall and component scores."""
    score = float(result["final_score"])
    st.metric("Overall similarity", f"{score:.1%}")
    st.progress(score)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pitch intervals", f"{float(result['interval_similarity']):.1%}")
    col2.metric("Note duration", f"{float(result['duration_similarity']):.1%}")
    col3.metric("Melodic contour", f"{float(result['contour_similarity']):.1%}")
    col4.metric("Beat placement", f"{float(result['beat_similarity']):.1%}")


def main() -> None:
    st.set_page_config(page_title="MelodyMatch", page_icon="🎵", layout="wide")
    st.title("MelodyMatch")
    st.write(
        "Compare symbolic melodies and see how pitch movement, rhythm, contour, "
        "and beat placement shape their similarity."
    )

    try:
        features = cached_features()
    except Exception as exc:
        st.error(f"Could not load the demonstration melodies: {exc}")
        st.stop()

    song_ids = sorted(features)
    st.subheader("Choose two melodies")
    left, right = st.columns(2)
    song_a = left.selectbox("First melody", song_ids, index=0)
    song_b = right.selectbox("Second melody", song_ids, index=1)

    if song_a == song_b:
        st.warning("Choose two different melodies to compare.")
        st.stop()

    result = compare_selected_pair(features, song_a, song_b)
    show_similarity_metric_grid(result)

    with st.expander("How this pair differs"):
        details = {
            "Note-count difference": int(result["note_count_difference"]),
            "Average-pitch difference": round(float(result["average_pitch_difference"]), 2),
            "Average-duration difference": round(float(result["average_duration_difference"]), 3),
            "Pitch-range difference": int(result["pitch_range_difference"]),
        }
        st.dataframe(details, use_container_width=True)

    st.subheader("All demonstration comparisons")
    st.dataframe(compare_all_pairs(features), use_container_width=True)

    with st.expander("Loaded melody details"):
        st.dataframe(feature_summary_frame(features), use_container_width=True)

    st.caption("Demonstration melodies are excerpts from the POP909 dataset.")


if __name__ == "__main__":
    main()
