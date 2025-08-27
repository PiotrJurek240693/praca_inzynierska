import time
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from code.game_communication.DistanceDetector import DistanceDetector
from code.game_communication.VehicleDataClient import VehicleDataClient
from code.game_communication.WindowCapture import WindowCapture
from code.environment.MathHelper import MathHelper

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
        self.minus_reward_counter = 0

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
                x1, y1, x2, y2, typ = map(float, line.strip().split())
                checkpoints.append(((x1, y1), (x2, y2), typ))
        return checkpoints

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.controller.reset_keys()

        #if self.win or self.resets_to_last_checkpoint >= 10:
        self.controller.press_key("=")
        time.sleep(0.1)
        self.controller.release_key("=")
        time.sleep(1.2)
        self.current_checkpoint = 0
        self.resets_to_last_checkpoint = 0
        self.last_checkpoint_index = -1
        #else:
        #    self.controller.press_key("backspace")
        #    time.sleep(0.1)
        #    self.controller.release_key("backspace")
        #    time.sleep(1.2)
        #    self.current_checkpoint = self.last_checkpoint_index + 1
        #    self.resets_to_last_checkpoint += 1

        self.previous_position = None
        self.previous_speed = None
        self.checkpoint_timer_start = time.time()
        self.win = False
        self.pause_time = time.time()
        self.flip = not self.flip
        self.minus_reward_counter = 0

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
        self.win = False if self.data_client.get_value("HasFinished") == 0 else True
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
        rays = obs[:-1]
        speed = obs[-1]

        if self.previous_position and self.current_checkpoint < len(self.checkpoints):
            start, end, typ = self.checkpoints[self.current_checkpoint]
            previous_distance = MathHelper.distance(self.previous_position, start, end)
            current_distance = MathHelper.distance(position, start, end)

            if MathHelper.cross_line(self.previous_position, position, start, end):
                reward += previous_distance
                if typ == 1:
                    self.last_checkpoint_index = self.current_checkpoint
                    self.resets_to_last_checkpoint = 0
                elif typ == 2:
                    done = True
                    print("done")
                    return reward, done

                #start_next, end_next, _ = self.checkpoints[self.current_checkpoint + 1]
                #max_distance = MathHelper.distance_point_to_line(
                #    MathHelper.two_points_midpoint(start, end),
                #    start_next,
                #    end_next
                #)
                #current_distance = MathHelper.distance_point_to_line(position, start_next, end_next)
                #reward += max_distance - current_distance

                self.current_checkpoint += 1
                self.checkpoint_timer_start = time.time()
            else:
                reward += previous_distance - current_distance

        min_distance = float(np.min(rays))
        if reward > 0:
            reward *= (min_distance ** 2) * 10
            self.minus_reward_counter = 0
        else:
            self.minus_reward_counter += 1

        if self.minus_reward_counter > 30:
            done = True
        if time.time() - self.checkpoint_timer_start > 20.0:
            done = True
        if self.previous_speed and self.previous_speed - speed > 25:
            done = True
        if min_distance < 0.01:
            done = True

        self.previous_speed = speed
        self.previous_position = position

        #print(reward)

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
