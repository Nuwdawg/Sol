"""Build the canonical 1024 px cache for MORDEN-style mask/spine training."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from shared.data_split import assert_disjoint, make_split


NATIVE = 2048


def image_channels(raw: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    background = cv2.GaussianBlur(resized, (0, 0), sigmaX=max(3.0, size / 96.0))
    dark_contrast = np.clip((background.astype(np.float32) - resized) * 4.0 + 128.0, 0, 255)
    return np.stack([resized, clahe, dark_contrast.astype(np.uint8)])


def targets(annotations: list[dict], size: int, spine_width: int) -> tuple[np.ndarray, np.ndarray]:
    scale = size / NATIVE
    mask = np.zeros((size, size), np.uint8)
    spine = np.zeros((size, size), np.uint8)
    for annotation in annotations:
        segmentation = annotation.get("segmentation") or []
        for ring in segmentation:
            if len(ring) >= 6:
                points = np.rint(np.asarray(ring).reshape(-1, 2) * scale).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)
        line = annotation.get("spine") or []
        if len(line) >= 4:
            points = np.rint(np.asarray(line).reshape(-1, 2) * scale).astype(np.int32)
            cv2.polylines(spine, [points], False, 1, thickness=spine_width)
    return mask, spine


def build(annotations_path: str, images_dir: str, output: str, size: int, spine_width: int) -> None:
    coco = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    records = {record["id"]: record for record in coco["images"]}
    by_image: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    split = make_split(annotations_path)
    assert_disjoint(split)
    print(split.summary(), flush=True)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    for fold, image_ids, stems in (
        ("train", split.train_image_ids, split.train_stems),
        ("val", split.val_image_ids, split.val_stems),
    ):
        stem_index = {stem: index for index, stem in enumerate(stems)}
        images = np.lib.format.open_memmap(
            out / f"{fold}_images.npy", mode="w+", dtype=np.uint8,
            shape=(len(stems), 3, size, size),
        )
        for index, stem in enumerate(stems):
            raw = cv2.imread(str(Path(images_dir) / stem), cv2.IMREAD_GRAYSCALE)
            if raw is None or raw.shape != (NATIVE, NATIVE):
                raise ValueError(f"invalid photograph {stem}: {None if raw is None else raw.shape}")
            images[index] = image_channels(raw, size)
            if (index + 1) % 100 == 0 or index + 1 == len(stems):
                print(f"{fold} photographs {index + 1}/{len(stems)}", flush=True)
        images.flush()

        masks = np.lib.format.open_memmap(
            out / f"{fold}_masks.npy", mode="w+", dtype=np.uint8,
            shape=(len(image_ids), size, size),
        )
        spines = np.lib.format.open_memmap(
            out / f"{fold}_spines.npy", mode="w+", dtype=np.uint8,
            shape=(len(image_ids), size, size),
        )
        record_to_photo = np.empty(len(image_ids), np.int32)
        for index, image_id in enumerate(image_ids):
            record = records[image_id]
            masks[index], spines[index] = targets(by_image[image_id], size, spine_width)
            record_to_photo[index] = stem_index[record["file_name"]]
            if (index + 1) % 200 == 0 or index + 1 == len(image_ids):
                print(f"{fold} targets {index + 1}/{len(image_ids)}", flush=True)
        masks.flush()
        spines.flush()
        np.save(out / f"{fold}_record_to_photo.npy", record_to_photo)
        (out / f"{fold}_ids.json").write_text(json.dumps(image_ids), encoding="utf-8")
        (out / f"{fold}_stems.json").write_text(json.dumps(stems), encoding="utf-8")
        print(
            f"{fold}: {len(stems)} photographs, {len(image_ids)} records, "
            f"mask={float(masks.mean()):.5f}, spine={float(spines.mean()):.5f}",
            flush=True,
        )

    manifest = {
        "size": size,
        "spine_width": spine_width,
        "split": split.summary(),
        "channels": ["raw", "clahe", "dark_local_contrast"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--spine-width", type=int, default=5)
    args = parser.parse_args()
    build(args.annotations, args.images, args.output, args.size, args.spine_width)


if __name__ == "__main__":
    main()
