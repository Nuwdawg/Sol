"""Convergence-bound training control: decay on plateau, stop only when converged.

Training length is decided by the validation signal, never by a clock or a fixed
epoch count. After warmup the learning rate stays at lr0. When the smoothed
validation fitness has not improved by more than `min_delta` for `patience`
epochs, the learning rate is multiplied by `decay_factor` and the plateau counter
restarts, so every lower rate gets a full window to prove itself. Training stops
only when a plateau persists after `max_decays` decays. That is the operational
meaning of "converged" in this experiment: more epochs, at every learning rate
tried, stopped improving validation.

Why not the stock Ultralytics schedule
--------------------------------------
Its cosine or linear schedule is a function of the epoch count, so the epoch count
has to be chosen before training starts. exp_010 chose 400, ran out of session at
62 with the rate still at its peak, and never consolidated. Early stopping on a
fixed schedule has the opposite failure: it can stop before the rate has annealed.
Decay on plateau removes both, because the rate only falls when the data says the
current rate has stopped helping.

Why smoothing
-------------
Validation here is noisy: 180 annotation records from multiple annotators who
often disagree. A single lucky epoch should not reset patience, and a single
unlucky one should not trigger a decay, so decisions use a moving average.

The controller is pure. It takes (completed_epochs, fitness) and returns a
decision; the Ultralytics callback that applies it lives in `train.py`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PlateauConfig:
    patience: int = 12
    min_delta: float = 0.002
    decay_factor: float = 0.2
    max_decays: int = 3
    smoothing: int = 3
    warmup_epochs: int = 3

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be at least 1 epoch")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative")
        if not 0.0 < self.decay_factor < 1.0:
            raise ValueError("decay_factor must lie strictly between 0 and 1")
        if self.max_decays < 0:
            raise ValueError("max_decays cannot be negative")
        if self.smoothing < 1:
            raise ValueError("smoothing must be at least 1")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative")


@dataclass
class Decision:
    epoch: int
    fitness: float
    smoothed: float | None
    best_smoothed: float | None
    best_epoch: int | None
    epochs_since_best: int
    decays: int
    lr_scale: float
    action: str
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


class PlateauController:
    """Decides, once per validated epoch, whether to continue, decay or stop."""

    def __init__(self, config: PlateauConfig | None = None):
        self.config = config or PlateauConfig()
        self._window: deque[float] = deque(maxlen=self.config.smoothing)
        self.best_smoothed: float | None = None
        self.best_epoch: int | None = None
        self.epochs_since_best = 0
        self.decays = 0
        self.lr_scale = 1.0
        self.history: list[Decision] = []

    def update(self, epoch: int, fitness: float) -> Decision:
        """Record the fitness of a completed epoch (1-based) and decide."""
        if fitness is None or not math.isfinite(float(fitness)):
            raise ValueError(f"fitness must be a finite number, got {fitness!r}")
        if self.history and epoch <= self.history[-1].epoch:
            raise ValueError(f"epochs must increase: {epoch} after {self.history[-1].epoch}")
        config = self.config
        self._window.append(float(fitness))
        smoothed = (sum(self._window) / len(self._window)
                    if len(self._window) == config.smoothing else None)

        action = "continue"
        if epoch <= config.warmup_epochs:
            reason = "warmup"
        elif smoothed is None:
            reason = "filling smoothing window"
        elif self.best_smoothed is None or smoothed > self.best_smoothed + config.min_delta:
            self.best_smoothed = smoothed
            self.best_epoch = epoch
            self.epochs_since_best = 0
            reason = "improved"
        else:
            self.epochs_since_best += 1
            if self.epochs_since_best < config.patience:
                reason = "plateau counting"
            elif self.decays < config.max_decays:
                self.decays += 1
                self.lr_scale *= config.decay_factor
                self.epochs_since_best = 0
                action = "decay"
                reason = (f"no smoothed improvement above {config.min_delta} "
                          f"for {config.patience} epochs")
            else:
                action = "stop"
                reason = (f"converged: plateau persisted for {config.patience} epochs "
                          f"after {config.max_decays} decays")

        decision = Decision(
            epoch=epoch, fitness=float(fitness), smoothed=smoothed,
            best_smoothed=self.best_smoothed, best_epoch=self.best_epoch,
            epochs_since_best=self.epochs_since_best, decays=self.decays,
            lr_scale=self.lr_scale, action=action, reason=reason)
        self.history.append(decision)
        return decision

    def state(self) -> dict:
        return dict(config=asdict(self.config), best_smoothed=self.best_smoothed,
                    best_epoch=self.best_epoch, epochs_since_best=self.epochs_since_best,
                    decays=self.decays, lr_scale=self.lr_scale,
                    window=list(self._window))
