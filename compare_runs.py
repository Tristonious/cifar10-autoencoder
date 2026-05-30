# Compare training curves across multiple history CSV files.
# Usage: python compare_runs.py history__<tag1>.csv history__<tag2>.csv --metric loss --save figures/compare.png

import argparse
from src.viz import compare_training_curves

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay training curves from multiple history CSVs.")
    parser.add_argument("files", nargs="+", help="One or more history__<tag>.csv files")
    parser.add_argument("--metric", default="loss", help="Metric to plot (e.g. loss, mae, mse)")
    parser.add_argument("--save", default=None, help="Path to save the output PNG")
    args = parser.parse_args()

    compare_training_curves(args.files, metric=args.metric, save_path=args.save)
