"""One-line JSON events on stdout, mirrored to a log file.

The pod's container log is the only live channel back from the run: the RunPod
MCP tools read it, and they cannot open a shell or copy files. Everything worth
knowing is therefore printed as `TAG {json}` on a single line, while the verbose
Ultralytics output goes to a file that is uploaded with the weights.
"""
from __future__ import annotations

import json
from pathlib import Path


def emit(tag: str, payload: dict, log_path: Path | None = None) -> str:
    line = f"{tag} {json.dumps(payload, sort_keys=True, default=str)}"
    print(line, flush=True)
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return line
