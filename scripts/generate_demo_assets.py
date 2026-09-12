"""Generate reproducible recruiter-facing results from the bundled demos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from melodymatch.pipeline import load_selected_features
from melodymatch.similarity import compare_all_pairs


PROJECT_ROOT = Path(__file__).parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"


def main() -> None:
    features = load_selected_features(PROJECT_ROOT / "data" / "POP909")
    results = compare_all_pairs(features).sort_values("final_score")
    results["pair"] = results["song_a"] + " vs " + results["song_b"]

    DOCS_DIR.mkdir(exist_ok=True)
    results.to_csv(DOCS_DIR / "demo_results.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10, 5.6))
    bars = axis.barh(results["pair"], results["final_score"] * 100, color="#7c3aed")
    axis.bar_label(bars, fmt="%.1f%%", padding=5, fontsize=9)
    axis.set_xlim(75, 90)
    axis.set_xlabel("Overall similarity (%)")
    axis.set_ylabel("Demo melody pair")
    figure.suptitle(
        "MelodyMatch demo comparisons",
        x=0.12,
        y=0.97,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    figure.text(
        0.12,
        0.91,
        "Weighted DTW similarity across pitch intervals, duration, contour, and beat placement",
        fontsize=9,
        color="#475569",
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.tight_layout(rect=[0, 0, 1, 0.86])
    figure.savefig(DOCS_DIR / "demo_similarity.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
