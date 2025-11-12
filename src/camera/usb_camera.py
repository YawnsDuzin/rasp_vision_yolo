"""
USB Camera interface using OpenCV
"""

import cv2
import logging
from threading import Thread, Lock


class USBCamera:
    """USB Camera handler with threading support"""

    def __init__(self, device_id=0, width=640, height=480, fps=30):
        """
        Initialize USB camera

        Args:
            device_id: Camera device ID (default: 0 for /dev/video0)
            width: Frame width
            height: Frame height
            fps: Target frames per second
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = Lock()

        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.device_id}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Use MJPEG format for better performance
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        # Read first frame
        self.ret, self.frame = self.cap.read()

        self.logger.info(f"USB Camera started: {self.width}x{self.height} @ {self.fps}fps")

        return self

    def start_threaded(self):
        """Start camera in separate thread for better performance"""
        self.start()
        Thread(target=self._update_frame, daemon=True).start()
        return self

    def _update_frame(self):
        """Update frame in background thread"""
        while not self.stopped:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                self.stopped = True

    def read(self):
        """
        Read current frame

        Returns:
            tuple: (ret, frame) where ret is success flag
        """
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Stop camera capture"""
        self.stopped = True
        if self.cap is not None:
            self.cap.release()
        self.logger.info("USB Camera stopped")

    def __enter__(self):
        """Context manager entry"""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


if __name__ == "__main__":
    # Test USB camera
    logging.basicConfig(level=logging.INFO)

    with USBCamera(device_id=0, width=640, height=480) as cam:
        print("Press 'q' to quit")

        while True:
            ret, frame = cam.read()

            if not ret or frame is None:
                print("Failed to read frame")
                break

            cv2.imshow('USB Camera Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
