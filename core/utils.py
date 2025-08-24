from __future__ import annotations
import math
from typing import Tuple

def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n <= 0:
        return (0.0, 0.0, 0.0)
    phat = successes / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    margin = (z * ((phat*(1-phat) + z*z/(4*n)) / n) ** 0.5) / denom
    return phat, max(0.0, center - margin), min(1.0, center + margin)
