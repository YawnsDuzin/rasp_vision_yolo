"""
YOLO Object Detector
"""

import cv2
import logging
from pathlib import Path
from ultralytics import YOLO


class YOLODetector:
    """YOLO-based object detector"""

    def __init__(
        self,
        model_path='yolov8n.pt',
        conf_threshold=0.5,
        iou_threshold=0.45,
        classes=None,
        device='cpu'
    ):
        """
        Initialize YOLO detector

        Args:
            model_path: Path to YOLO model file
            conf_threshold: Confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS
            classes: List of class IDs to detect (None for all)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.device = device

        self.logger = logging.getLogger(__name__)

        # Load model
        self.model = self._load_model()

        self.logger.info(
            f"YOLO model loaded: {model_path} "
            f"(conf={conf_threshold}, iou={iou_threshold})"
        )

    def _load_model(self):
        """Load YOLO model"""
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def detect(self, frame, conf=None, iou=None, classes=None):
        """
        Detect objects in frame

        Args:
            frame: Input image (numpy array)
            conf: Confidence threshold (override default)
            iou: IoU threshold (override default)
            classes: Class IDs to detect (override default)

        Returns:
            Ultralytics Results object
        """
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold
        classes = classes if classes is not None else self.classes

        results = self.model(
            frame,
            conf=conf,
            iou=iou,
            classes=classes,
            verbose=False
        )

        return results[0]

    def draw_detections(self, frame, results, show_labels=True, show_conf=True):
        """
        Draw detection boxes on frame

        Args:
            frame: Input image
            results: Detection results
            show_labels: Whether to show class labels
            show_conf: Whether to show confidence scores

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return annotated_frame

        for box in boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Get class name
            class_name = results.names[cls]

            # Draw bounding box
            color = self._get_color(cls)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Create label
            if show_labels:
                label_parts = []
                if show_labels:
                    label_parts.append(class_name)
                if show_conf:
                    label_parts.append(f"{conf:.2f}")

                label = " ".join(label_parts)

                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        return annotated_frame

    def _get_color(self, class_id):
        """Get color for class ID"""
        # Generate consistent colors based on class ID
        import random
        random.seed(class_id)
        return tuple(random.randint(0, 255) for _ in range(3))

    def get_detection_info(self, results):
        """
        Extract detection information

        Args:
            results: Detection results

        Returns:
            List of detection dictionaries
        """
        detections = []

        boxes = results.boxes
        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detection = {
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'confidence': conf,
                'class_id': cls,
                'class_name': results.names[cls]
            }

            detections.append(detection)

        return detections


if __name__ == "__main__":
    # Test YOLO detector
    import sys

    logging.basicConfig(level=logging.INFO)

    # Check if image path provided
    if len(sys.argv) < 2:
        print("Usage: python yolo_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)

    # Initialize detector
    detector = YOLODetector(model_path='yolov8n.pt', conf_threshold=0.5)

    # Detect objects
    results = detector.detect(frame)

    # Draw detections
    annotated = detector.draw_detections(frame, results)

    # Get detection info
    detections = detector.get_detection_info(results)
    print(f"\nDetected {len(detections)} objects:")
    for det in detections:
        print(f"  - {det['class_name']}: {det['confidence']:.2f}")

    # Display result
    cv2.imshow('YOLO Detection', annotated)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
