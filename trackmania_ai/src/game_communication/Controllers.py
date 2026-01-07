import keyboard
import vgamepad as vg

class KeyboardController:
    _MIN_VALUE = 0.3

    def __init__(self) -> None:
        self.pressed_keys = set()

    def press_key(self, key: str) -> None:
        pressed_key = key.lower()
        if pressed_key not in self.pressed_keys:
            keyboard.press(pressed_key)
            self.pressed_keys.add(pressed_key)

    def release_key(self, key: str) -> None:
        pressed_key = key.lower()
        if pressed_key in self.pressed_keys:
            keyboard.release(pressed_key)
            self.pressed_keys.remove(pressed_key)

    def reset_keys(self) -> None:
        for pressed_key in list(self.pressed_keys):
            keyboard.release(pressed_key)
        self.pressed_keys.clear()


    def steer(self, steer: float, throttle: float) -> None:
        if steer > self._MIN_VALUE:
            self.release_key("a")
            self.press_key("d")
        elif steer < -self._MIN_VALUE:
            self.release_key("d")
            self.press_key("a")
        else:
            self.release_key("a")
            self.release_key("d")

        if throttle > self._MIN_VALUE:
            self.release_key("s")
            self.press_key("w")
        elif throttle < -self._MIN_VALUE:
            self.release_key("w")
            self.press_key("s")
        else:
            self.release_key("w")
            self.release_key("s")


class GamepadController:
    def __init__(self) -> None:
        self.pressed_keys = set()
        self._pad = vg.VX360Gamepad()

    def press_key(self, key: str) -> None:
        pressed_key = key.lower()
        if pressed_key not in self.pressed_keys:
            keyboard.press(pressed_key)
            self.pressed_keys.add(pressed_key)

    def release_key(self, key: str) -> None:
        pressed_key = key.lower()
        if pressed_key in self.pressed_keys:
            keyboard.release(pressed_key)
            self.pressed_keys.remove(pressed_key)

    def reset_keys(self) -> None:
        for pressed_key in list(self.pressed_keys):
            keyboard.release(pressed_key)
        self.pressed_keys.clear()
        self._pad.left_joystick_float(x_value_float=0, y_value_float=0)
        self._pad.right_trigger_float(value_float=0)
        self._pad.left_trigger_float(value_float=0)
        self._pad.update()

    def steer(self, steer: float, throttle: float) -> None:
        self._pad.left_joystick_float(x_value_float=steer, y_value_float=0)
        if throttle > 0.0:
            self._pad.right_trigger_float(value_float=1)
            self._pad.left_trigger_float(value_float=0)
        else:
            self._pad.right_trigger_float(value_float=0)
            self._pad.left_trigger_float(value_float=1)
        self._pad.update()