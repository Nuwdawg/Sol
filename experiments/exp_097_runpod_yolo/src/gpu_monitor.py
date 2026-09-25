"""Sample GPU utilization, memory and power with nvidia-smi on a background thread.

The requirement for this run is sustained utilization above 70%. Utilization is
the fraction of sampling time in which a kernel was executing, so it can read
high while the GPU does little; power draw against the board limit is recorded
beside it as the harder evidence of real work. Every sample is appended to a CSV
that is uploaded with the weights, so the claim can be checked afterwards.
"""
from __future__ import annotations

import csv
from pathlib import Path
import statistics
import subprocess
import threading
import time
from typing import Callable

QUERY = ("utilization.gpu", "memory.used", "memory.total", "power.draw",
         "power.limit", "temperature.gpu")
COLUMNS = ("utilization", "memory_used_mib", "memory_total_mib", "power_w",
           "power_limit_w", "temperature_c")


def parse_line(line: str) -> dict | None:
    """Parse one `--format=csv,noheader,nounits` row; unsupported fields become None."""
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) != len(QUERY):
        return None
    sample = {}
    for column, value in zip(COLUMNS, parts):
        try:
            sample[column] = float(value)
        except ValueError:
            sample[column] = None
    return sample if sample["utilization"] is not None else None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    position = (len(ordered) - 1) * fraction
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class GPUMonitor(threading.Thread):
    def __init__(self, csv_path: Path | None, interval: float = 2.0, index: int = 0,
                 clock: Callable[[], float] = time.time):
        super().__init__(daemon=True, name="gpu-monitor")
        self.csv_path = Path(csv_path) if csv_path else None
        self.interval = interval
        self.index = index
        self._clock = clock
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.samples: list[tuple[float, dict]] = []
        self.errors = 0

    def command(self) -> list[str]:
        return ["nvidia-smi", f"--id={self.index}", f"--query-gpu={','.join(QUERY)}",
                "--format=csv,noheader,nounits"]

    def record(self, timestamp: float, sample: dict) -> None:
        with self._lock:
            self.samples.append((timestamp, sample))
        if self.csv_path is not None:
            new = not self.csv_path.exists()
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("a", newline="") as handle:
                writer = csv.writer(handle)
                if new:
                    writer.writerow(("time",) + COLUMNS)
                writer.writerow([f"{timestamp:.3f}"] + [sample[c] for c in COLUMNS])

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                output = subprocess.run(self.command(), capture_output=True, text=True,
                                        timeout=15, check=True).stdout
                sample = parse_line(output.splitlines()[0]) if output.strip() else None
                if sample is None:
                    self.errors += 1
                else:
                    self.record(self._clock(), sample)
            except (subprocess.SubprocessError, OSError, IndexError):
                self.errors += 1
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()

    def summarize(self, start: float, end: float | None = None) -> dict:
        with self._lock:
            window = [s for t, s in self.samples
                      if t >= start and (end is None or t <= end)]
        if not window:
            return dict(samples=0, errors=self.errors)
        utilization = [s["utilization"] for s in window]
        power = [s["power_w"] for s in window if s["power_w"] is not None]
        limits = [s["power_limit_w"] for s in window if s["power_limit_w"] is not None]
        memory = [s["memory_used_mib"] for s in window if s["memory_used_mib"] is not None]
        totals = [s["memory_total_mib"] for s in window if s["memory_total_mib"] is not None]
        return dict(
            samples=len(window), errors=self.errors,
            utilization_mean=statistics.fmean(utilization),
            utilization_p10=percentile(utilization, 0.10),
            utilization_p50=percentile(utilization, 0.50),
            power_mean_w=statistics.fmean(power) if power else None,
            power_limit_w=max(limits) if limits else None,
            power_fraction=(statistics.fmean(power) / max(limits)) if power and limits else None,
            memory_max_mib=max(memory) if memory else None,
            memory_total_mib=max(totals) if totals else None)
