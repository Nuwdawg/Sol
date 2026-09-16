"""Remove submission instances below an exact decoded area threshold."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from pycocotools import mask as mu

from scripts.submit import validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--min-area", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["filament_id", "segmentation_rle"]
        source = list(reader)
    kept = []
    for row in source:
        rle = dict(size=[2048, 2048], counts=row["segmentation_rle"])
        if int(mu.area(rle)) >= args.min_area:
            kept.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(kept)
    validate(args.output)
    report = dict(input=str(args.input), min_area=args.min_area,
                  input_rows=len(source), rows=len(kept), removed=len(source) - len(kept),
                  sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
