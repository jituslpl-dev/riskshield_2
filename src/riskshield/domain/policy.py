from dataclasses import dataclass
import numpy as np


def decisions(p, threshold, review_threshold=None):
    lower = threshold / 2 if review_threshold is None else review_threshold
    return np.where(p >= threshold, "decline", np.where(p >= lower, "review", "approve"))


@dataclass(frozen=True)
class CostPolicy:
    """Strategy: minimize expected cost, including imperfect manual review."""

    missed_event_cost: float = 100.0
    false_decline_cost: float = 5.0
    review_cost: float = 1.0
    review_sensitivity: float = 0.95
    review_specificity: float = 0.99

    def __post_init__(self):
        values = [
            self.missed_event_cost,
            self.false_decline_cost,
            self.review_cost,
            self.review_sensitivity,
            self.review_specificity,
        ]
        if not all(np.isfinite(values)) or min(values[:3]) < 0:
            raise ValueError("Costs must be finite and nonnegative.")
        if not all(0 <= x <= 1 for x in values[3:]):
            raise ValueError("Review accuracy must be in [0,1].")

    def evaluate(self, probability):
        p = np.asarray(probability, dtype=float)
        if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("Probabilities must be finite and in [0,1].")
        costs = np.column_stack(
            [
                p * self.missed_event_cost,
                np.full(len(p), self.review_cost)
                + p * (1 - self.review_sensitivity) * self.missed_event_cost
                + (1 - p) * (1 - self.review_specificity) * self.false_decline_cost,
                (1 - p) * self.false_decline_cost,
            ]
        )
        index = costs.argmin(axis=1)
        return np.array(["approve", "review", "decline"])[index], costs
