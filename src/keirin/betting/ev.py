"""Expected value computation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidatePick:
    combo: str
    prob: float
    odds: float

    @property
    def ev(self) -> float:
        return self.prob * self.odds


def build_candidates(
    probs: dict[str, float],
    odds: dict[str, float],
) -> list[CandidatePick]:
    out: list[CandidatePick] = []
    for combo, prob in probs.items():
        o = odds.get(combo)
        if o is None or o <= 0:
            continue
        out.append(CandidatePick(combo=combo, prob=prob, odds=o))
    return out
