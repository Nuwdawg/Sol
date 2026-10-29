"""Rebuild the model checkpoints from the parts stored in this folder, and verify them.

GitHub rejects files over 100 MB, so checkpoints larger than 45 MiB are stored as numbered parts.
`MANIFEST.json` lists, for every checkpoint, its SHA256, its origin, and the SHA256 of each part.
This script checks every part, concatenates them into `weights/assembled/<name>`, and checks the
whole file again. Any mismatch stops it.

    python weights/assemble.py            # writes weights/assembled/*.pt
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def assemble(out_dir: Path = HERE / "assembled") -> dict[str, Path]:
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, entry in manifest["files"].items():
        target = out_dir / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"]:
            paths[name] = target
            continue
        digest = hashlib.sha256()
        with target.open("wb") as handle:
            for part in entry["parts"]:
                data = (HERE / part["file"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != part["sha256"]:
                    raise SystemExit(f"{part['file']}: SHA256 differs from MANIFEST.json")
                handle.write(data)
                digest.update(data)
        if digest.hexdigest() != entry["sha256"]:
            target.unlink()
            raise SystemExit(f"{name}: assembled file does not match MANIFEST.json")
        paths[name] = target
        print(f"{name}: {entry['bytes']:,} bytes, SHA256 {entry['sha256'][:16]}... verified")
    return paths


if __name__ == "__main__":
    assemble(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "assembled")
