"""
Camera test script
Tests different camera types (CSI, USB, RTSP)
"""

import cv2
import sys
import time
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, '/home/user/rasp_vision_yolo')

from src.camera import USBCamera, RTSPCamera, CSICamera


def main():
    parser = argparse.ArgumentParser(description='Camera test utility')
    parser.add_argument(
        '--camera',
        type=str,
        choices=['usb', 'csi', 'rtsp'],
        default='usb',
        help='Camera type to test'
    )
    parser.add_argument('--device', type=int, default=0, help='USB camera device ID')
    parser.add_argument('--url', type=str, help='RTSP stream URL')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize camera based on type
    camera = None

    try:
        if args.camera == 'usb':
            print(f"Testing USB camera (device {args.device})...")
            camera = USBCamera(
                device_id=args.device,
                width=args.width,
                height=args.height,
                fps=args.fps
            ).start_threaded()

        elif args.camera == 'csi':
            if CSICamera is None:
                print("Error: CSI camera not available. Install picamera2:")
                print("  sudo apt install python3-picamera2")
                return

            print("Testing CSI camera...")
            camera = CSICamera(
                width=args.width,
                height=args.height,
                fps=args.fps
            ).start()

        elif args.camera == 'rtsp':
            if not args.url:
                print("Error: --url required for RTSP camera")
                print("Example: --url rtsp://admin:password@192.168.1.100:554/stream")
                return

            print(f"Testing RTSP camera ({args.url})...")
            camera = RTSPCamera(rtsp_url=args.url).start_threaded()

        # FPS counter
        fps_counter = 0
        fps_start_time = time.time()
        fps_value = 0

        print("\nCamera started successfully!")
        print("Press 'q' to quit")
        print("Press 's' to save snapshot")
        print("-" * 50)

        while True:
            ret, frame = camera.read()

            if not ret or frame is None:
                print("Waiting for frames...")
                time.sleep(0.1)
                continue

            # Update FPS counter
            fps_counter += 1
            elapsed = time.time() - fps_start_time

            if elapsed > 1.0:
                fps_value = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()

            # Draw FPS on frame
            cv2.putText(
                frame,
                f"FPS: {fps_value:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Draw camera info
            camera_info = f"{args.camera.upper()} Camera - {args.width}x{args.height}"
            cv2.putText(
                frame,
                camera_info,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1
            )

            # Display frame
            cv2.imshow('Camera Test', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\nSnapshot saved: {filename}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if camera:
            camera.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")


if __name__ == "__main__":
    main()
