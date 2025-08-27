import numpy as np
from typing import Tuple

class MathHelper:
    @staticmethod
    def ccw(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float]
    ) -> bool:
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

    @staticmethod
    def cross_line(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float]
    ) -> bool:
        return (
            MathHelper.ccw(p1, a, b) != MathHelper.ccw(p2, a, b)
            and MathHelper.ccw(p1, p2, a) != MathHelper.ccw(p1, p2, b)
        )

    @staticmethod
    def distance(
        p: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float]
    ) -> float:
        p_array = np.array(p)
        a_array = np.array(a)
        b_array = np.array(b)
        ap = p_array - a_array
        ab = b_array - a_array
        return float(abs(np.cross(ab, ap)) / np.linalg.norm(ab))

    @staticmethod
    def midpoint(
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> Tuple[float, float]:
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)