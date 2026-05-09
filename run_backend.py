"""Command-line runner for MelodyMatch."""

from __future__ import annotations

import argparse

from melodymatch.config import DEFAULT_SONG_IDS, default_dataset_root
from melodymatch.pipeline import (
    feature_summary_frame,
    run_similarity_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare POP909 melodies and export pairwise similarity scores."
    )
    parser.add_argument(
        "--dataset-root",
        default=default_dataset_root(),
        help="Path to the POP909 dataset root.",
    )
    parser.add_argument(
        "--songs",
        nargs="+",
        default=list(DEFAULT_SONG_IDS),
        help="Selected POP909 song ids. Default: 001 002 003 004 005.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for CSV exports.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    features, pairwise, similarity_csv = run_similarity_pipeline(
        dataset_root=args.dataset_root,
        song_ids=args.songs,
        output_dir=args.output_dir,
    )

    print("\nLoaded melody tracks:")
    print(feature_summary_frame(features).to_string(index=False))

    print("\nPairwise similarity results:")
    print(pairwise.to_string(index=False))
    print(f"\nSimilarity CSV written to: {similarity_csv}")


if __name__ == "__main__":
    main()
