"""Command-line runner for the XMIDI genre-classification extension."""

from __future__ import annotations

import argparse

from melodymatch.xmidi.pipeline import run_xmidi_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an explainable XMIDI symbolic MIDI genre classifier."
    )
    parser.add_argument(
        "--dataset-root",
        default="XMIDI_Dataset",
        help="Path to the local XMIDI_Dataset folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for feature, prediction, and model outputs.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for faster smoke tests. Omit to use all files.",
    )
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse outputs/xmidi_features.csv instead of re-extracting features.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the feature dataset reserved for testing.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Decision tree max_depth.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and model training.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the random forest baseline report.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    def show_progress(index: int, total: int, filename: str) -> None:
        if index == 1 or index == total or index % 250 == 0:
            print(f"Extracting XMIDI features {index}/{total}: {filename}")

    result = run_xmidi_training_pipeline(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        max_files=args.max_files,
        reuse_features=args.reuse_features,
        test_size=args.test_size,
        max_depth=args.max_depth,
        random_state=args.random_state,
        report_baseline=not args.no_baseline,
        progress_callback=show_progress if not args.reuse_features else None,
    )

    print("\nXMIDI genre classifier evaluation:")
    print(result.evaluation_text)
    print("\nDecision tree rules:")
    print(result.tree_text)
    print("\nOutputs written:")
    print(f"Features: {result.feature_csv}")
    print(f"Test predictions: {result.predictions_csv}")
    print(f"Confusion matrix: {result.confusion_matrix_csv}")
    print(f"Saved model: {result.model_path}")


if __name__ == "__main__":
    main()

