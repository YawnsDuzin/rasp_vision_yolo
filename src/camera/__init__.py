"""
Camera module for Raspberry Pi YOLO Vision Project
Supports CSI, USB, and RTSP cameras
"""

from .usb_camera import USBCamera
from .rtsp_camera import RTSPCamera

try:
    from .csi_camera import CSICamera
except ImportError:
    # picamera2 not available
    CSICamera = None

__all__ = ['USBCamera', 'RTSPCamera', 'CSICamera']
