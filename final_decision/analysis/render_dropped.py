"""Native-resolution crops of instances exp169 drops, for visual review (CPU).

Validation panels: the dropped exp139 instance (yellow), every annotator's filaments (one colour
per record: green, cyan, magenta), exp169's own instances (red). Test panels: dropped exp139
instance (yellow) and exp169's instances (red); no labels exist or are used for test.
Images are read from the competition archive with HTTP range requests (one member at a time).
argv: archive_url annotations veto_check.json removed_instances.json base_csv new_csv out_dir
"""
import csv
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parent))   # httpzip.py sits beside this script
from httpzip import RangeFile  # noqa: E402

url, annotations, veto_path, removed_path, base_csv, new_csv, out_dir = sys.argv[1:8]
out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
z = zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=1 << 20))

def image(name, split):
    data = z.read(f"MAGFiLO_1.0_Kaggle_2026/{split}/{split}_images/{name}")
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)

coco = json.load(open(annotations))
by_file = defaultdict(list)
for r in coco["images"]:
    by_file[r["file_name"]].append(r["id"])
anns = defaultdict(list)
for a in coco["annotations"]:
    anns[a["image_id"]].append(a)

def load_csv(path):
    by = defaultdict(list)
    for row in csv.DictReader(open(path)):
        by[row["filament_id"].rsplit("_", 1)[0]].append({"size": [2048, 2048], "counts": row["segmentation_rle"].encode()})
    return by

base, new = load_csv(base_csv), load_csv(new_csv)

def outline(canvas, mask, colour, thickness=2):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, colour, thickness)

def crop_box(mask, pad=160):
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + pad
    y0, y1 = max(0, cy - half), min(2048, cy + half)
    x0, x1 = max(0, cx - half), min(2048, cx + half)
    return y0, y1, x0, x1

COLOURS = [(0, 200, 0), (255, 255, 0), (255, 0, 255)]

# validation: rebuild the dropped exp139 instance from the exp147 cache is not needed; we draw the
# exp139 candidate covering it, stored in veto_check's audit by photo/area only, so re-derive below.
audit = json.load(open(veto_path))["audit"]
val_panels = []
cache = json.load(open(sys.argv[8]))["photographs"] if len(sys.argv) > 8 else None
for item in sorted([a for a in audit if a["area"] >= 2000], key=lambda a: -a["area"])[:6]:
    name = item["photo"]
    img = image(name, "train")
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # exp139 deployed instance of that area: repaint exp139's rule for this photo
    cands = sorted([c for c in cache[name]["full"] if c["score"] >= 0.85], key=lambda c: -c["score"])
    claimed = np.zeros((2048, 2048), np.uint8); target = None
    for c in cands:
        m = mu.decode({"size": c["rle"]["size"], "counts": c["rle"]["counts"].encode()}) & (1 - claimed)
        if m.sum() < 100:
            continue
        claimed |= m
        if int(m.sum()) == item["area"]:
            target = m
    if target is None:
        continue
    for k, image_id in enumerate(sorted(by_file[name])):
        for a in anns[image_id]:
            rle = mu.merge(mu.frPyObjects(a["segmentation"], 2048, 2048))
            outline(canvas, mu.decode(rle), COLOURS[k % 3], 2)
    outline(canvas, target, (0, 255, 255), 3)
    y0, y1, x0, x1 = crop_box(target)
    panel = canvas[y0:y1, x0:x1].copy()
    label = f"VAL {name[:-5]} area {item['area']} s={item['score']:.2f} matched {item['tp_records']}/{item['records']}"
    cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    val_panels.append(panel)

removed = json.load(open(removed_path))
test_panels = []
for item in sorted([r for r in removed if r["status"] == "deleted"], key=lambda r: -r["area"])[:6]:
    stem = item["stem"]
    img = image(stem + ".jpeg", "test")
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    target = None
    for r in base[stem]:
        m = mu.decode(r)
        if int(m.sum()) == item["area"]:
            target = m
    if target is None:
        continue
    for r in new.get(stem, []):
        outline(canvas, mu.decode(r), (0, 0, 255), 2)
    outline(canvas, target, (0, 255, 255), 3)
    y0, y1, x0, x1 = crop_box(target)
    panel = canvas[y0:y1, x0:x1].copy()
    cv2.putText(panel, f"TEST {stem} area {item['area']} s={item['score']:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    test_panels.append(panel)

def sheet(panels, path, size=640, cols=3):
    tiles = [cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA) if p.shape[0] > size else
             cv2.copyMakeBorder(p, 0, size - p.shape[0], 0, size - p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
    rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))
    print("WROTE", path, len(panels))

sheet(val_panels, out / "val_dropped_large.jpg")
sheet(test_panels, out / "test_dropped_large.jpg")
