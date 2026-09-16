"""CPU-only validation of three existing-weight inference changes; no training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops
from experiments.exp_007_tta.src.dihedral import TRANSFORMS, apply_transform, invert_transform
from experiments.exp_033_inference.src.sweep import encode, records_for, evaluate_thresholds
from experiments.exp_033_inference.src.cached_tta import fuse

BASE_SHA = 'd9c6641edb1fec427212d0c35b008c923361e9fcf5004905a69c883714d93389'
# Native CPU decoding at proposal floor .25. Do not reuse the .05-floor
# historical scalar: crop_mask switches rounding at 50 retained detections.
# Provenance: scripts/audit_existing_weight_control.py and its saved report.
IDENTITY_CONTROL_PQ = .44145370603652173


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def crop_bounds(box, shape, total):
    # Match Ultralytics' CPU branch, including its n>=50 float-box path.
    h, w = shape
    coords = box.round() if total < 50 else box.ceil()
    x1, y1, x2, y2 = map(int, coords.tolist())
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def decode_planes(proto, coefficients, boxes, shape):
    c, h, w = proto.shape
    logits = (coefficients @ proto.float().view(c, -1)).view(-1, h, w)
    for i in range(len(boxes)):
        plane = ops.scale_masks(logits[i:i+1][None], shape)[0, 0]
        x1, y1, x2, y2 = crop_bounds(boxes[i], shape, len(boxes))
        binary = np.zeros(shape, np.uint8)
        probabilities = np.zeros(shape, np.float32)
        if x2 > x1 and y2 > y1:
            crop = plane[y1:y2, x1:x2]
            binary[y1:y2, x1:x2] = (crop > 0).cpu().numpy().astype(np.uint8)
            probabilities[y1:y2, x1:x2] = crop.sigmoid().cpu().numpy()
        yield binary, probabilities


class CapturePredictor(SegmentationPredictor):
    """Stream native masks, retaining only RLE and optionally a soft crop."""
    keep_soft = False
    transform_spec = (0, False)

    def construct_result(self, pred, img, orig_img, img_path, proto):
        self.captured = []
        if len(pred):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            for row, (binary, prob) in zip(pred, decode_planes(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])):
                if binary.sum() < 40:
                    continue
                k, mirror = self.transform_spec
                binary = invert_transform(binary, k, mirror)
                entry = dict(score=float(row[4]), rle=encode(binary))
                if self.keep_soft:
                    prob = invert_transform(prob, k, mirror)
                    ys, xs = np.nonzero(prob)
                    y1, y2, x1, x2 = int(ys.min()), int(ys.max()+1), int(xs.min()), int(xs.max()+1)
                    entry['crop'] = (x1, y1, x2, y2, prob[y1:y2, x1:x2].copy())
                self.captured.append(entry)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])


def infer(model, image, conf=.25, iou=.6, max_det=100, soft=False, transform=(0, False), device='cpu'):
    CapturePredictor.keep_soft = soft
    CapturePredictor.transform_spec = transform
    with torch.inference_mode():
        model.predict(apply_transform(image, *transform), predictor=CapturePredictor,
                      imgsz=2048, conf=conf, iou=iou, max_det=max_det,
                      retina_masks=True, device=device, verbose=False)
    if len(model.predictor.results[0].boxes) >= max_det:
        raise RuntimeError(f'Proposal cap {max_det} saturated; experiment is incomplete')
    return model.predictor.captured


def mask_nms(entries, threshold):
    if not entries:
        return []
    order = sorted(range(len(entries)), key=lambda i: -entries[i]['score'])
    ious = mu.iou([e['rle'] for e in entries], [e['rle'] for e in entries], [0]*len(entries))
    keep = []
    for i in order:
        if all(ious[i, j] <= threshold for j in keep):
            keep.append(i)
    return [entries[i] for i in keep]


def soft_fuse(entries, thresholds=(.4, .5, .6)):
    """Identical seed clusters and scores to historical hard D4 fusion."""
    output = {str(t): [] for t in thresholds}
    if not entries:
        return output
    order = sorted(range(len(entries)), key=lambda i: -entries[i]['score'])
    ious = mu.iou([e['rle'] for e in entries], [e['rle'] for e in entries], [0]*len(entries))
    used = set()
    for seed in order:
        if seed in used:
            continue
        members = [seed]; used.add(seed)
        for i in order:
            if i not in used and ious[seed, i] >= .5:
                members.append(i); used.add(i)
        score = float(np.mean([entries[i]['score'] for i in members]))*(.5+.5*len(members)/8)
        crops = [entries[i]['crop'] for i in members]
        x1, y1 = min(c[0] for c in crops), min(c[1] for c in crops)
        x2, y2 = max(c[2] for c in crops), max(c[3] for c in crops)
        average = np.zeros((y2-y1, x2-x1), np.float32)
        for a, b, c, d, values in crops:
            average[b-y1:d-y1, a-x1:c-x1] += values
        average /= len(members)
        for t in thresholds:
            binary = np.zeros(entries[seed]['rle']['size'], np.uint8)
            binary[y1:y2, x1:x2] = average > t
            if binary.sum() >= 40:
                output[str(t)].append(dict(score=score, rle=encode(binary)))
    return output


def interpolate(base, donor, alpha):
    if base.keys() != donor.keys() or any(base[k].shape != donor[k].shape for k in base):
        raise ValueError('Incompatible checkpoint state dictionaries')
    if not 0 <= alpha <= 1:
        raise ValueError(alpha)
    return {k: (v.clone() if alpha == 0 or not v.is_floating_point() else
                donor[k].clone() if alpha == 1 else torch.lerp(v, donor[k], alpha))
            for k, v in base.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['soft', 'nms', 'weights'], required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--donor')
    p.add_argument('--annotations', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    if sha != BASE_SHA:
        raise RuntimeError(f'Wrong baseline checkpoint: {sha}')
    records, split = records_for(args.annotations)
    if args.limit:
        # Distributed deterministic screen; never describe this as full validation.
        names = sorted(records)[::max(1, len(records)//args.limit)][:args.limit]
        records = {k: records[k] for k in names}
    report = dict(mode=args.mode, device='cpu', training=False, checkpoint_sha256=sha,
                  donor_sha256=hashlib.sha256(Path(args.donor).read_bytes()).hexdigest() if args.donor else None,
                  annotation_sha256=hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
                  split=split.summary(), photographs=len(records), full_validation=len(records)==106,
                  imgsz=2048, conf_floor=.25, variants={}, status='running')
    dump(root/'resolved.json', report)
    print('RESOLVED', json.dumps(report), flush=True)
    started = time.monotonic()
    model = YOLO(args.weights)
    model.model.eval().requires_grad_(False)
    caches = {}

    def image_for(name):
        gray = cv2.imread(str(Path(args.images)/name), cv2.IMREAD_GRAYSCALE)
        if gray is not None and gray.ndim == 3 and gray.shape[2] == 1:
            gray = gray[:, :, 0]
        if gray is None or gray.shape != (2048, 2048):
            raise ValueError(name)
        return np.repeat(gray[:, :, None], 3, axis=2)

    def record_progress(variant, n):
        state = dict(variant=variant, completed=n, total=len(records), seconds=time.monotonic()-started)
        dump(root/'progress.json', state)
        print('PROGRESS', json.dumps(state), flush=True)

    if args.mode == 'weights':
        donor = YOLO(args.donor).model.float().eval().requires_grad_(False)
        if model.model.names != donor.names:
            raise ValueError('Class mappings differ')
        base_state = {k: v.detach().clone() for k, v in model.model.float().state_dict().items()}
        donor_state = donor.state_dict()
        if not all(torch.isfinite(v).all() for v in donor_state.values()):
            raise ValueError('Nonfinite donor')
        del model
        for alpha in (0., .25, .5, .75, 1.):
            variant = f'alpha_{alpha}'
            # A fresh unfused model is essential: predict() fuses BatchNorm in place.
            model = YOLO(args.weights)
            model.model.load_state_dict(interpolate(base_state, donor_state, alpha), strict=True)
            model.model.eval().requires_grad_(False)
            cache = {}
            for i, name in enumerate(sorted(records)):
                cache[name] = infer(model, image_for(name))
                record_progress(variant, i+1)
            caches[variant] = cache
            dump(root/f'{variant}.json', cache)
            del model
    else:
        variants = ['hard_control', 'soft_0.4', 'soft_0.5', 'soft_0.6'] if args.mode == 'soft' else ['box_control', 'mask_0.3', 'mask_0.5', 'mask_0.7']
        caches = {v: {} for v in variants}
        for i, name in enumerate(sorted(records)):
            image = image_for(name)
            if args.mode == 'soft':
                entries = []
                for transform in TRANSFORMS:
                    entries.extend(infer(model, image, soft=True, transform=transform))
                caches['hard_control'][name] = fuse([(e['score'], e['rle']) for e in entries], .5, .5)
                for threshold, fused in soft_fuse(entries).items():
                    caches[f'soft_{threshold}'][name] = fused
            else:
                caches['box_control'][name] = infer(model, image)
                entries = infer(model, image, iou=1., max_det=1000)
                for threshold in (.3, .5, .7):
                    caches[f'mask_{threshold}'][name] = mask_nms(entries, threshold)
            record_progress(args.mode, i+1)
            if (i+1) % 10 == 0:
                for variant, cache in caches.items():
                    dump(root/f'{variant}.json', cache)
        for variant, cache in caches.items():
            dump(root/f'{variant}.json', cache)
    for variant, cache in caches.items():
        rows = []
        for grow in (-1, 0):
            rows.extend(evaluate_thresholds(cache, records, (.30, .35, .385, .40, .45, .50), grow=grow))
        report['variants'][variant] = dict(sweep=rows, best=max(rows, key=lambda r: r['pq']))
        print('RESULT', variant, json.dumps(report['variants'][variant]['best']), flush=True)
        dump(root/'report.json', report)
    control = {'soft':'hard_control', 'nms':'box_control', 'weights':'alpha_0.0'}[args.mode]
    report['control'] = control
    fixed_conf = .40 if args.mode == 'soft' else .35
    expected = .45707595553359726 if args.mode == 'soft' else IDENTITY_CONTROL_PQ
    tolerance = .001 if args.mode == 'soft' else 1e-6
    actual = next(r['pq'] for r in report['variants'][control]['sweep'] if r['conf']==fixed_conf and r['grow']==-1)
    report['historical_control'] = dict(expected=expected, actual=actual,
                                        tolerance=tolerance,
                                        passed=abs(actual-expected)<tolerance if len(records)==106 else None)
    report['best_variant'] = max(report['variants'], key=lambda k:report['variants'][k]['best']['pq'])
    report['seconds'] = time.monotonic()-started
    report['status'] = 'complete'
    dump(root/'report.json', report)
    if report['historical_control']['passed'] is False:
        report['status'] = 'control_mismatch'
        dump(root/'report.json', report)
        raise RuntimeError(f'Historical control mismatch: {actual} versus {expected}')
    if hashlib.sha256(Path(args.weights).read_bytes()).hexdigest() != sha:
        raise RuntimeError('Source checkpoint changed')


if __name__ == '__main__':
    main()
