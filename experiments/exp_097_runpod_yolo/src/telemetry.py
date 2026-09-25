"""Per-epoch throughput accounting that separates waiting for data from GPU work.

GPU utilization alone cannot say *why* a GPU is idle. The usual culprit at 2048 px
is the input pipeline: JPEG decode and affine augmentation of 4-megapixel images
on too few CPU workers. This splits every training iteration into

- data wait: from the end of one batch to the start of the next, which is time
  the training loop spent blocked on the dataloader; and
- step time: from batch start to batch end, which is forward, loss, backward and
  (every `accumulate` batches) the optimizer step.

CUDA kernels run asynchronously, so a CPU timestamp can close a step before the
GPU has finished it, and the unpaid work then shows up as data wait. Calling
`synchronize` every `sync_every` batches keeps the attribution honest on average
at a negligible cost for iterations that each take a large fraction of a second.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable


@dataclass
class EpochSummary:
    batches: int
    images: int
    seconds: float
    images_per_second: float
    data_wait_seconds: float
    data_wait_fraction: float
    first_batch_wait_seconds: float
    step_seconds: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class EpochTelemetry:
    def __init__(self, clock: Callable[[], float] = time.perf_counter,
                 synchronize: Callable[[], None] | None = None, sync_every: int = 25):
        if sync_every < 1:
            raise ValueError("sync_every must be at least 1")
        self._clock = clock
        self._synchronize = synchronize
        self._sync_every = sync_every
        self._reset()

    def _reset(self) -> None:
        self._epoch_start: float | None = None
        self._batch_start: float | None = None
        self._last_end: float | None = None
        self._pending_images = 0
        self.batches = 0
        self.images = 0
        self.data_wait = 0.0
        self.first_batch_wait = 0.0
        self.step_time = 0.0

    def epoch_start(self) -> None:
        self._reset()
        self._epoch_start = self._clock()

    def batch_start(self, batch_images: int) -> None:
        if self._epoch_start is None:
            self.epoch_start()
        now = self._clock()
        if self._last_end is None:
            # The first fetch includes worker start-up; keep it out of the steady ratio.
            self.first_batch_wait = now - self._epoch_start
        else:
            self.data_wait += now - self._last_end
        self._batch_start = now
        self._pending_images = int(batch_images)

    def batch_end(self) -> None:
        if self._batch_start is None:
            raise RuntimeError("batch_end called before batch_start")
        if self._synchronize is not None and (self.batches + 1) % self._sync_every == 0:
            self._synchronize()
        now = self._clock()
        self.step_time += now - self._batch_start
        self.images += self._pending_images
        self.batches += 1
        self._last_end = now
        self._batch_start = None

    def summary(self) -> EpochSummary:
        if self._epoch_start is None or self._last_end is None:
            return EpochSummary(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        seconds = self._last_end - self._epoch_start
        steady = max(seconds - self.first_batch_wait, 1e-9)
        return EpochSummary(
            batches=self.batches, images=self.images, seconds=seconds,
            images_per_second=self.images / max(seconds, 1e-9),
            data_wait_seconds=self.data_wait,
            data_wait_fraction=self.data_wait / steady,
            first_batch_wait_seconds=self.first_batch_wait,
            step_seconds=self.step_time)
