"""
CSI Camera interface using picamera2 (Raspberry Pi Camera Module)
"""

import logging
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    logging.warning("picamera2 not available. Install with: sudo apt install python3-picamera2")


class CSICamera:
    """CSI Camera handler for Raspberry Pi Camera Module"""

    def __init__(self, width=640, height=480, fps=30):
        """
        Initialize CSI camera

        Args:
            width: Frame width
            height: Frame height
            fps: Target frames per second
        """
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "picamera2 is not installed. "
                "Install with: sudo apt install python3-picamera2"
            )

        self.width = width
        self.height = height
        self.fps = fps

        self.picam2 = None
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start CSI camera"""
        self.picam2 = Picamera2()

        # Configure camera
        config = self.picam2.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps}
        )

        self.picam2.configure(config)
        self.picam2.start()

        self.logger.info(f"CSI Camera started: {self.width}x{self.height} @ {self.fps}fps")

        return self

    def read(self):
        """
        Read current frame

        Returns:
            tuple: (ret, frame) where ret is success flag and frame is in BGR format
        """
        try:
            # Capture frame in RGB888 format
            frame_rgb = self.picam2.capture_array()

            # Convert RGB to BGR for OpenCV compatibility
            import cv2
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            return True, frame_bgr
        except Exception as e:
            self.logger.error(f"Failed to capture frame: {e}")
            return False, None

    def stop(self):
        """Stop CSI camera"""
        if self.picam2 is not None:
            self.picam2.stop()
        self.logger.info("CSI Camera stopped")

    def __enter__(self):
        """Context manager entry"""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


# Legacy picamera support
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    import time

    class CSICameraLegacy:
        """Legacy CSI Camera handler using picamera (for older systems)"""

        def __init__(self, width=640, height=480, fps=30):
            self.width = width
            self.height = height
            self.fps = fps

            self.camera = None
            self.raw_capture = None
            self.logger = logging.getLogger(__name__)

        def start(self):
            """Start CSI camera"""
            self.camera = PiCamera()
            self.camera.resolution = (self.width, self.height)
            self.camera.framerate = self.fps

            self.raw_capture = PiRGBArray(self.camera, size=(self.width, self.height))

            # Allow camera to warm up
            time.sleep(0.1)

            self.logger.info(f"CSI Camera (legacy) started: {self.width}x{self.height}")
            return self

        def read(self):
            """Read current frame"""
            try:
                self.raw_capture.truncate(0)
                self.camera.capture(self.raw_capture, format="bgr")
                frame = self.raw_capture.array
                return True, frame
            except Exception as e:
                self.logger.error(f"Failed to capture frame: {e}")
                return False, None

        def stop(self):
            """Stop CSI camera"""
            if self.camera is not None:
                self.camera.close()
            self.logger.info("CSI Camera (legacy) stopped")

        def __enter__(self):
            return self.start()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop()

except ImportError:
    CSICameraLegacy = None


if __name__ == "__main__":
    # Test CSI camera
    import cv2

    logging.basicConfig(level=logging.INFO)

    print("Testing CSI camera...")
    print("Press 'q' to quit")

    with CSICamera(width=640, height=480, fps=30) as cam:
        while True:
            ret, frame = cam.read()

            if not ret or frame is None:
                print("Failed to read frame")
                break

            cv2.imshow('CSI Camera Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
