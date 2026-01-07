import time
from typing import Any, Dict, List, Optional, Tuple
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.game_communication.DistanceDetector import DistanceDetector
from src.game_communication.VehicleDataClient import VehicleDataClient
from src.game_communication.WindowCapture import WindowCapture

class TrackmaniaEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(
        self,
        controller,
        checkpoint_file: str
    ) -> None:
        super().__init__()

        self.controller = controller
        self.data_client = VehicleDataClient()
        self.window_capture = WindowCapture("Trackmania")
        self.ray_detector = DistanceDetector()

        self.checkpoints = self.load_checkpoints(checkpoint_file)

        (self.centerline_points,
         self.segment_lengths,
         self.cumulative_lengths,
         self.segment_half_widths,
         self.total_track_length) = self._build_centerline(self.checkpoints)

        self.previous_s: Optional[float] = None

        self.previous_position: Optional[Tuple[float, float]] = None
        self.previous_speed: Optional[float] = None
        self.checkpoint_timer_start = time.time()
        self.last_checkpoint_index = -1
        self.resets_to_last_checkpoint = 0
        self.current_checkpoint = 0
        self.win = False
        self.pause_time = time.time()
        self.paused = False
        self.flip = False
        self.too_slow_counter = 0

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(38,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.data_client.connect()

    def load_checkpoints(self, filepath: str) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        checkpoints: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.replace(',', '.')
                x1, y1, x2, y2, typ = map(float, line.strip().split())
                checkpoints.append(((x1, y1), (x2, y2), typ))
        return checkpoints

    def _build_centerline(
            self,
            checkpoints: List[Tuple[Tuple[float, float], Tuple[float, float], float]]
    ):
        centerline_points: List[Tuple[float, float]] = []
        half_width_points: List[float] = []

        for (x1, y1), (x2, y2), _typ in checkpoints:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            width = math.dist((x1, y1), (x2, y2))
            half_width = 0.5 * width

            if centerline_points:
                lastx, lasty = centerline_points[-1]
                if (cx - lastx) ** 2 + (cy - lasty) ** 2 < 1e-4:
                    continue

            centerline_points.append((cx, cy))
            half_width_points.append(half_width)

        segment_lengths = []
        cumulative_lengths = [0.0]
        segment_half_widths = []

        for i in range(len(centerline_points) - 1):
            x1, y1 = centerline_points[i]
            x2, y2 = centerline_points[i + 1]
            seg_len = math.dist((x1, y1), (x2, y2))
            segment_lengths.append(seg_len)
            cumulative_lengths.append(cumulative_lengths[-1] + seg_len)

            hw = min(half_width_points[i], half_width_points[i + 1])
            segment_half_widths.append(hw)

        segment_lengths_arr = np.array(segment_lengths, dtype=np.float32)
        cumulative_lengths_arr = np.array(cumulative_lengths, dtype=np.float32)
        segment_half_widths_arr = np.array(segment_half_widths, dtype=np.float32)
        total_length = float(cumulative_lengths_arr[-1])

        return (
            centerline_points,
            segment_lengths_arr,
            cumulative_lengths_arr,
            segment_half_widths_arr,
            total_length,
        )

    def _project_onto_centerline(self, position: Tuple[float, float]) -> Tuple[float, int]:
        px, py = position
        best_s = 0.0
        best_dist_sq = float("inf")
        best_idx = 0

        for i in range(len(self.centerline_points) - 1):
            ax, ay = self.centerline_points[i]
            bx, by = self.centerline_points[i + 1]

            vx = bx - ax
            vy = by - ay
            wx = px - ax
            wy = py - ay

            seg_len_sq = vx * vx + vy * vy
            if seg_len_sq <= 1e-8:
                continue

            t = (wx * vx + wy * vy) / seg_len_sq
            t = max(0.0, min(1.0, t))

            proj_x = ax + t * vx
            proj_y = ay + t * vy

            dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                seg_len = math.sqrt(seg_len_sq)
                best_s = float(self.cumulative_lengths[i] + seg_len * t)
                best_idx = i

        return best_s, best_idx

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.controller.reset_keys()

        self.controller.press_key("=")
        time.sleep(0.1)
        self.controller.release_key("=")
        time.sleep(1.2)

        self.current_checkpoint = 0
        self.resets_to_last_checkpoint = 0
        self.last_checkpoint_index = -1

        self.previous_position = None
        self.previous_speed = None
        self.previous_s = None
        self.checkpoint_timer_start = time.time()
        self.win = False
        self.pause_time = time.time()
        self.flip = not self.flip
        self.too_slow_counter = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        steer, throttle = float(action[0]), float(action[1])
        if self.flip:
            steer *= -1.0

        self.controller.steer(steer=steer, throttle=throttle)

        while self.data_client.semaphore.acquire(blocking=False):
            pass
        self.data_client.semaphore.acquire(timeout=0.1)

        obs = self._get_obs()
        position = (self.data_client.get_value("Position.x"), self.data_client.get_value("Position.z"))

        reward, done = self._compute_reward(obs, position)

        self.win = self.data_client.get_value("HasFinished") != 0
        if self.win:
            print("HasFinished")
            done = True

        return obs, reward, done, False, {}

    def _get_obs(self) -> np.ndarray:
        img = self.window_capture.capture_window()
        rays = np.array(self.ray_detector.get_distances(img), dtype=np.float32)
        if self.flip:
            rays = np.flip(rays)
        speed = np.array([self.data_client.get_value("Speed") * 0.001], dtype=np.float32)
        return np.concatenate([rays, speed], axis=0)

    def _compute_reward(self, obs: np.ndarray, position: Tuple[float, float]) -> Tuple[float, bool]:
        done = False
        reward = 0.0

        speed = float(obs[-1])

        s_current, seg_idx = self._project_onto_centerline(position)

        if self.previous_s is None:
            delta_s = 0.0
        else:
            delta_s = s_current - self.previous_s

        if self.total_track_length > 0:
            progress_increment = delta_s / self.total_track_length
        else:
            progress_increment = 0.0

        px, py = position
        cx0, cy0 = self.centerline_points[seg_idx]
        cx1, cy1 = self.centerline_points[seg_idx + 1]

        vx = cx1 - cx0
        vy = cy1 - cy0
        wx = px - cx0
        wy = py - cy0

        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq > 1e-8:
            t = (wx * vx + wy * vy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = cx0 + t * vx
            proj_y = cy0 + t * vy
            distance_from_center = math.dist((px, py), (proj_x, proj_y))
        else:
            distance_from_center = 0.0

        half_width = float(self.segment_half_widths[seg_idx])

        if distance_from_center > half_width - 4.5:
            done = True

        reward_progress = progress_increment * 100.0

        if reward_progress > 0:
            center_factor = max(0.0, 1.0 - (distance_from_center / half_width))
            reward_progress *= center_factor

        reward += reward_progress

        if speed < 0.02:
            self.too_slow_counter += 1
        else:
            self.too_slow_counter = 0

        if self.too_slow_counter > 100:
            done = True

        if time.time() - self.checkpoint_timer_start > 60.0:
            done = True

        self.previous_speed = speed
        self.previous_position = position
        self.previous_s = s_current

        return reward, done

    def stop(self) -> None:
        if self.paused is False:
            self.pause_time = time.time()
            self.controller.reset_keys()
            self.controller.press_key('esc')
            time.sleep(0.1)
            self.controller.release_key("esc")
            self.paused = True

    def resume(self) -> None:
        if self.paused is True:
            self.checkpoint_timer_start += time.time() - self.pause_time
            self.controller.reset_keys()
            self.controller.press_key('esc')
            time.sleep(0.1)
            self.controller.release_key("esc")
            self.paused = False
