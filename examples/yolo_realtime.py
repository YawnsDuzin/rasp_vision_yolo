"""
Real-time YOLO object detection with camera
"""

import cv2
import sys
import time
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, '/home/user/rasp_vision_yolo')

from src.camera import USBCamera, RTSPCamera, CSICamera
from src.detection import YOLODetector


def main():
    parser = argparse.ArgumentParser(description='Real-time YOLO object detection')
    parser.add_argument('--camera', type=str, choices=['usb', 'csi', 'rtsp'], default='usb')
    parser.add_argument('--device', type=int, default=0, help='USB camera device ID')
    parser.add_argument('--url', type=str, help='RTSP stream URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--classes', type=int, nargs='+', help='Class IDs to detect')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--imgsz', type=int, default=640, help='YOLO input size')
    parser.add_argument('--skip-frames', type=int, default=1, help='Process every N frames')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize camera
    camera = None

    try:
        print(f"Initializing {args.camera.upper()} camera...")

        if args.camera == 'usb':
            camera = USBCamera(
                device_id=args.device,
                width=args.width,
                height=args.height
            ).start_threaded()

        elif args.camera == 'csi':
            if CSICamera is None:
                print("Error: CSI camera not available")
                return
            camera = CSICamera(width=args.width, height=args.height).start()

        elif args.camera == 'rtsp':
            if not args.url:
                print("Error: --url required for RTSP")
                return
            camera = RTSPCamera(rtsp_url=args.url).start_threaded()

        # Initialize YOLO detector
        print(f"Loading YOLO model: {args.model}...")
        detector = YOLODetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            classes=args.classes
        )

        # Warm up model
        print("Warming up model...")
        dummy_frame = cv2.imread('/dev/null') or cv2.VideoCapture(0).read()[1]
        if dummy_frame is not None:
            _ = detector.detect(dummy_frame[:args.imgsz, :args.imgsz])

        print("\nStarting real-time detection!")
        print("Press 'q' to quit")
        print("Press 's' to save detection snapshot")
        print("-" * 50)

        # Performance metrics
        frame_counter = 0
        processed_frames = 0
        fps_counter = 0
        fps_start = time.time()
        inference_times = []

        while True:
            ret, frame = camera.read()

            if not ret or frame is None:
                print("Waiting for frames...")
                time.sleep(0.1)
                continue

            frame_counter += 1

            # Frame skipping for performance
            if frame_counter % args.skip_frames != 0:
                continue

            # Resize for YOLO
            h, w = frame.shape[:2]
            if max(h, w) != args.imgsz:
                scale = args.imgsz / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h))
            else:
                resized = frame

            # Run detection
            start_time = time.time()
            results = detector.detect(resized)
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)

            # Draw detections on original frame
            annotated = detector.draw_detections(frame, results)

            # Get detection info
            detections = detector.get_detection_info(results)
            processed_frames += 1

            # Calculate FPS
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed > 1.0:
                fps = fps_counter / elapsed
                avg_inference = sum(inference_times) / len(inference_times)

                # Print stats
                print(
                    f"FPS: {fps:.1f} | "
                    f"Inference: {avg_inference:.1f}ms | "
                    f"Detections: {len(detections)}"
                )

                fps_counter = 0
                fps_start = time.time()
                inference_times = []

            # Draw stats on frame
            stats_text = [
                f"FPS: {fps:.1f}",
                f"Inference: {inference_time:.1f}ms",
                f"Objects: {len(detections)}"
            ]

            y_offset = 30
            for text in stats_text:
                cv2.putText(
                    annotated,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                y_offset += 30

            # Display
            cv2.imshow('YOLO Real-time Detection', annotated)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"\nSnapshot saved: {filename}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if camera:
            camera.stop()
        cv2.destroyAllWindows()

        print(f"\nProcessed {processed_frames} frames")
        print("Detection stopped")


if __name__ == "__main__":
    main()
