"""Poll one Kaggle competition submission until a public score is available."""
import argparse
import json
import time
from pathlib import Path

from kaggle import api

p = argparse.ArgumentParser()
p.add_argument("competition")
p.add_argument("ref")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)
while True:
    submission = next((x for x in api.competition_submissions(a.competition)
                       if str(x.ref) == a.ref), None)
    if submission is None:
        raise SystemExit("submission ref not found")
    state = {k: str(getattr(submission, k, None))
             for k in ("ref", "file_name", "status", "public_score", "description")}
    state["checked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    a.output.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state), flush=True)
    if any(x in state["status"].upper() for x in ("COMPLETE", "ERROR")):
        break
    time.sleep(60)
