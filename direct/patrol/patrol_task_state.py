"""[Patrol120 change] Task bookkeeping only; never generates robot actions."""
from dataclasses import dataclass
import math


@dataclass
class HoldTimer:
    tolerance: float = 0.01
    entered_at: float | None = None
    first_reach: float | None = None
    longest_hold: float = 0.0

    def update(self, elapsed, error):
        if not math.isfinite(error):
            self.entered_at = None
            return 0.0
        if error < self.tolerance:
            if self.first_reach is None:
                self.first_reach = elapsed
            if self.entered_at is None:
                self.entered_at = elapsed
            held = elapsed - self.entered_at
            self.longest_hold = max(self.longest_hold, held)
            return held
        self.entered_at = None
        return 0.0


@dataclass
class PatrolMonitor:
    """Observed FK alone advances tasks; truth only computes evaluation metrics."""
    hold_seconds: float = 1.0
    tolerance: float = 0.01

    def __post_init__(self):
        if self.hold_seconds < 0 or not math.isfinite(self.hold_seconds):
            raise ValueError('hold_seconds must be finite and nonnegative')
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError('tolerance must be finite and positive')
        self.observed = HoldTimer(self.tolerance)
        self.truth = HoldTimer(self.tolerance)

    def update(self, elapsed, observed_error, true_error):
        observed_hold = self.observed.update(elapsed, observed_error)
        self.truth.update(elapsed, true_error)
        # [Patrol120 change] Privileged truth cannot decide when to switch target.
        return (math.isfinite(observed_error) and observed_error < self.tolerance
                and observed_hold + 1e-9 >= self.hold_seconds)

    def report(self):
        return {
            'true_first_reach_elapsed_s': self.truth.first_reach,
            'observed_first_reach_elapsed_s': self.observed.first_reach,
            'true_longest_continuous_hold_s': self.truth.longest_hold,
            'observed_longest_continuous_hold_s': self.observed.longest_hold,
            'true_hold_passed': (self.truth.first_reach is not None
                                 and self.truth.longest_hold + 1e-9 >= self.hold_seconds),
        }
