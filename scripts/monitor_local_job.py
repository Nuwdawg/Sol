"""Persist compact status for a background process without chat polling."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import time


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        # os.kill(pid, 0) calls TerminateProcess on Windows.  Query the process
        # handle instead so observing a worker cannot stop it.
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    args.status.parent.mkdir(parents=True, exist_ok=True)

    while True:
        alive = process_alive(args.pid)
        tail = []
        if args.log and args.log.exists():
            tail = args.log.read_text(errors="replace").splitlines()[-12:]
        payload = dict(pid=args.pid, alive=alive, checked_at=time.time(), log_tail=tail)
        if args.report and args.report.exists():
            payload["report"] = json.loads(args.report.read_text())
        temporary = args.status.with_suffix(args.status.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(args.status)
        if not alive:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
