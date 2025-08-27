import math
import numpy as np
from typing import List, Optional, Tuple


class DistanceDetector:
    @staticmethod
    def _trace_ray(angle: float, gray: np.ndarray, width: int, height: int, start_x: int, start_y: int) -> Tuple[float, float]:
        distance = None
        end_x = start_x
        end_y = start_y
        angle_radian = math.radians(angle)
        dx = math.sin(angle_radian)
        dy = -math.cos(angle_radian)

        while 0 <= int(end_x) < width and 0 <= int(end_y) < height:
            if distance is None and gray[int(end_y), int(end_x)].item() <= 50:
                distance = math.hypot(end_x - start_x, end_y - start_y)
            end_x += dx
            end_y += dy

        max_distance = math.hypot(end_x - start_x, end_y - start_y)

        if distance is None:
            distance = max_distance

        return distance, max_distance

    def get_distances(self, image: np.ndarray) -> List[float]:
        height, width = image.shape
        distances: List[float] = []
        max_distances: List[Optional[float]] = []

        for angle in range(-90, 91, 5):
            distance, max_distance = self._trace_ray(angle, image, width, height, width // 2, height - 1)
            distances.append(distance)
            max_distances.append(max_distance)

        output = [distance / max_distance if max_distance > 0 else 0.0 for distance, max_distance in zip(distances, max_distances)]
        return output
