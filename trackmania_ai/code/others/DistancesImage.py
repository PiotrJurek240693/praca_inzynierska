import os
import cv2
import numpy as np
from code.game_communication.WindowCapture import WindowCapture
from code.game_communication.DistanceDetector import DistanceDetector

if __name__ == "__main__":
    capture = WindowCapture("Trackmania")
    detector = DistanceDetector()

    image = capture.capture_window()
    height, width = image.shape

    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for angle in range(-90, 91, 5):
        distance, max_distance = detector._trace_ray(
            angle, image, width, height, width // 2, height - 1
        )

        radians = np.radians(angle)
        dx = np.sin(radians)
        dy = -np.cos(radians)

        end_x = int(width // 2 + dx * distance)
        end_y = int(height - 1 + dy * distance)

        cv2.line(output, (width // 2, height - 1), (end_x, end_y), (0, 0, 255), 1)

    path = os.path.join("../../images", "distances12.png")
    cv2.imwrite(path, output)