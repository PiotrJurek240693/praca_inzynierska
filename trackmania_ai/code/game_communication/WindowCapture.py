import win32gui
import win32ui
import win32con
import numpy as np
from PIL import Image

class WindowCapture:
    _BORDER_OFFSET = 8
    _TITLEBAR_OFFSET = 31

    def __init__(self, window_title: str) -> None:
        self.window_title = window_title
        self.window = win32gui.FindWindow(None, self.window_title)

    def capture_window(self) -> np.ndarray:
        left, top, right, bottom = win32gui.GetWindowRect(self.window)
        width = right - left
        height = bottom - top

        window_dc = win32gui.GetWindowDC(self.window)
        dc = win32ui.CreateDCFromHandle(window_dc)
        compatible_dc = dc.CreateCompatibleDC()

        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(dc, width, height)
        compatible_dc.SelectObject(bitmap)
        compatible_dc.BitBlt((0, 0), (width, height), dc, (0, 0), win32con.SRCCOPY)

        bitmap_info = bitmap.GetInfo()
        bitmap_bits = bitmap.GetBitmapBits(True)

        image = Image.frombuffer('RGB', (bitmap_info['bmWidth'], bitmap_info['bmHeight']), bitmap_bits, 'raw', 'BGRX', 0, 1)
        image = image.crop((self._BORDER_OFFSET, self._TITLEBAR_OFFSET, image.width - self._BORDER_OFFSET, image.height - self._BORDER_OFFSET))
        image = image.convert('L')

        dc.DeleteDC()
        compatible_dc.DeleteDC()
        win32gui.ReleaseDC(self.window, window_dc)
        win32gui.DeleteObject(bitmap.GetHandle())

        return np.array(image)