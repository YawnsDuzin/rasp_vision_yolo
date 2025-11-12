"""
RTSP Network Camera interface
"""

import cv2
import logging
import time
from threading import Thread, Lock


class RTSPCamera:
    """RTSP camera handler with reconnection support"""

    def __init__(self, rtsp_url, buffer_size=1, reconnect_delay=5):
        """
        Initialize RTSP camera

        Args:
            rtsp_url: RTSP stream URL (e.g., rtsp://user:pass@ip:554/stream)
            buffer_size: Frame buffer size (1 for minimal latency)
            reconnect_delay: Delay between reconnection attempts (seconds)
        """
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay

        self.cap = None
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = Lock()

        self.logger = logging.getLogger(__name__)

    def _connect(self):
        """Establish RTSP connection"""
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if self.cap.isOpened():
            # Minimize buffer for reduced latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            return True
        return False

    def start(self):
        """Start RTSP stream"""
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            if self._connect():
                self.ret, self.frame = self.cap.read()
                self.logger.info(f"RTSP stream connected: {self.rtsp_url}")
                return self
            else:
                retry_count += 1
                self.logger.warning(
                    f"RTSP connection failed (attempt {retry_count}/{max_retries})"
                )
                time.sleep(self.reconnect_delay)

        raise RuntimeError(f"Cannot connect to RTSP stream: {self.rtsp_url}")

    def start_threaded(self):
        """Start RTSP stream in separate thread"""
        self.start()
        Thread(target=self._update_frame, daemon=True).start()
        return self

    def _update_frame(self):
        """Update frame in background thread with auto-reconnect"""
        consecutive_failures = 0
        max_failures = 30  # Reconnect after 30 consecutive failures

        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                self.logger.warning("RTSP stream disconnected, attempting reconnect...")
                if self._connect():
                    consecutive_failures = 0
                    self.logger.info("RTSP stream reconnected")
                else:
                    time.sleep(self.reconnect_delay)
                    continue

            ret, frame = self.cap.read()

            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                self.logger.warning(f"Frame read failed ({consecutive_failures})")

                if consecutive_failures >= max_failures:
                    self.logger.error("Too many failures, reconnecting...")
                    self.cap.release()
                    consecutive_failures = 0

    def read(self):
        """
        Read current frame

        Returns:
            tuple: (ret, frame) where ret is success flag
        """
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Stop RTSP stream"""
        self.stopped = True
        if self.cap is not None:
            self.cap.release()
        self.logger.info("RTSP stream stopped")

    def __enter__(self):
        """Context manager entry"""
        return self.start_threaded()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


if __name__ == "__main__":
    # Test RTSP camera
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python rtsp_camera.py <rtsp_url>")
        print("Example: python rtsp_camera.py rtsp://admin:password@192.168.1.100:554/stream")
        sys.exit(1)

    rtsp_url = sys.argv[1]

    with RTSPCamera(rtsp_url) as cam:
        print("Press 'q' to quit")

        while True:
            ret, frame = cam.read()

            if not ret or frame is None:
                print("Waiting for frames...")
                time.sleep(0.1)
                continue

            cv2.imshow('RTSP Camera Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
