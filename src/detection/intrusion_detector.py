"""
Intrusion Detection System
침입 감지 시스템 - 보안 및 감시 분야 활용
"""

import cv2
import numpy as np
import logging
from datetime import datetime
from typing import List, Tuple, Optional


class IntrusionDetector:
    """
    ROI(Region of Interest) 기반 침입 감지 시스템
    특정 영역에 사람이나 차량이 진입하면 알림 발생
    """

    def __init__(
        self,
        roi_points: List[Tuple[int, int]],
        target_classes: List[int] = None,
        cooldown_seconds: int = 5,
        min_confidence: float = 0.5
    ):
        """
        Initialize intrusion detector

        Args:
            roi_points: ROI 다각형 좌표 [(x1,y1), (x2,y2), ...]
            target_classes: 탐지할 클래스 ID 리스트 (None이면 모든 클래스)
            cooldown_seconds: 알림 간격 (초)
            min_confidence: 최소 신뢰도
        """
        self.roi_points = np.array(roi_points, dtype=np.int32)
        self.target_classes = target_classes
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence

        self.last_alert_time = None
        self.intrusion_count = 0
        self.current_intruders = set()

        self.logger = logging.getLogger(__name__)

    def check_intrusion(self, detections: List[dict]) -> Tuple[bool, List[dict]]:
        """
        침입 여부 확인

        Args:
            detections: 탐지된 객체 리스트

        Returns:
            (침입 여부, 침입 객체 리스트)
        """
        intruders = []
        current_ids = set()

        for det in detections:
            # 신뢰도 체크
            if det['confidence'] < self.min_confidence:
                continue

            # 클래스 필터링
            if self.target_classes and det['class_id'] not in self.target_classes:
                continue

            # ROI 내부 확인
            center = det['center']
            if self._point_in_roi(center):
                intruders.append(det)

                # 고유 ID가 있으면 추적
                if 'track_id' in det:
                    current_ids.add(det['track_id'])

        # 새로운 침입자 카운트
        new_intruders = current_ids - self.current_intruders
        if new_intruders:
            self.intrusion_count += len(new_intruders)
            self.logger.info(f"새로운 침입자 {len(new_intruders)}명 탐지")

        self.current_intruders = current_ids

        # 침입 발생 여부
        intrusion_detected = len(intruders) > 0

        return intrusion_detected, intruders

    def _point_in_roi(self, point: Tuple[int, int]) -> bool:
        """점이 ROI 내부에 있는지 확인"""
        result = cv2.pointPolygonTest(self.roi_points, point, False)
        return result >= 0

    def should_alert(self) -> bool:
        """쿨다운 시간 체크하여 알림 발생 여부 결정"""
        if self.last_alert_time is None:
            return True

        elapsed = (datetime.now() - self.last_alert_time).total_seconds()
        return elapsed >= self.cooldown_seconds

    def trigger_alert(self):
        """알림 트리거"""
        self.last_alert_time = datetime.now()
        self.logger.warning(f"침입 알림! (총 {self.intrusion_count}회)")

    def draw_roi(self, frame: np.ndarray, color=(0, 255, 255), thickness=2):
        """
        프레임에 ROI 그리기

        Args:
            frame: 입력 프레임
            color: ROI 선 색상 (BGR)
            thickness: 선 두께
        """
        # ROI 다각형 그리기
        cv2.polylines(frame, [self.roi_points], True, color, thickness)

        # 반투명 영역 표시
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.roi_points], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        return frame

    def draw_intruders(
        self,
        frame: np.ndarray,
        intruders: List[dict],
        color=(0, 0, 255)
    ):
        """
        침입자 표시

        Args:
            frame: 입력 프레임
            intruders: 침입자 리스트
            color: 박스 색상
        """
        for intruder in intruders:
            x1, y1, x2, y2 = intruder['bbox']

            # 빨간색 박스로 강조
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # 경고 레이블
            label = f"⚠ {intruder['class_name']} {intruder['confidence']:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # 중심점 표시
            center = intruder['center']
            cv2.circle(frame, center, 5, color, -1)

        return frame

    def get_statistics(self) -> dict:
        """통계 정보 반환"""
        return {
            'total_intrusions': self.intrusion_count,
            'current_intruders': len(self.current_intruders),
            'last_alert_time': self.last_alert_time
        }

    def reset_statistics(self):
        """통계 초기화"""
        self.intrusion_count = 0
        self.current_intruders.clear()
        self.last_alert_time = None


if __name__ == "__main__":
    # 테스트 코드
    import sys
    sys.path.insert(0, '/home/user/rasp_vision_yolo')

    from src.camera import USBCamera
    from src.detection import YOLODetector

    logging.basicConfig(level=logging.INFO)

    # ROI 설정 (화면 중앙 영역)
    roi = [(200, 150), (440, 150), (440, 330), (200, 330)]

    # 침입 감지기 초기화 (사람만 탐지)
    detector = IntrusionDetector(
        roi_points=roi,
        target_classes=[0],  # 0 = person
        cooldown_seconds=3
    )

    # 카메라 및 YOLO 초기화
    camera = USBCamera(width=640, height=480).start_threaded()
    yolo = YOLODetector(model_path='yolov8n.pt')

    print("침입 감지 시스템 시작")
    print("Press 'q' to quit, 'r' to reset statistics")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            # 객체 탐지
            results = yolo.detect(frame)
            detections = yolo.get_detection_info(results)

            # 침입 체크
            intrusion, intruders = detector.check_intrusion(detections)

            # 시각화
            frame = detector.draw_roi(frame)
            frame = yolo.draw_detections(frame, results)

            if intrusion:
                frame = detector.draw_intruders(frame, intruders)

                if detector.should_alert():
                    detector.trigger_alert()
                    print(f"⚠ 침입 감지! {len(intruders)}명")

            # 통계 표시
            stats = detector.get_statistics()
            cv2.putText(
                frame,
                f"Intrusions: {stats['total_intrusions']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow('Intrusion Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset_statistics()
                print("통계 초기화됨")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
