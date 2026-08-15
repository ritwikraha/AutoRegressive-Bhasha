from __future__ import annotations

import argparse

from ocn.detectors import OCNDetector
from ocn.io import read_table, write_table
from ocn.metrics import detection_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect OCN candidates in cached generations.")
    parser.add_argument("--input", required=True, help="Input CSV/JSON/JSONL file.")
    parser.add_argument("--output", required=True, help="Output CSV/JSON/JSONL file.")
    parser.add_argument("--text-column", default="response", help="Column containing model response text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_table(args.input)
    detector = OCNDetector()
    scored = detector.annotate_rows(df, text_column=args.text_column)
    write_table(scored, args.output)
    print(detection_summary(scored).to_string())


if __name__ == "__main__":
    main()
