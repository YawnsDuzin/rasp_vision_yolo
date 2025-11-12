"""
Line Crossing Counter
라인 통과 카운팅 - 스마트시티, 리테일 분야 활용
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class LineCrossingCounter:
    """
    라인 통과 카운팅 시스템
    사람, 차량 등이 특정 라인을 통과할 때 카운팅
    양방향 카운팅 지원
    """

    def __init__(
        self,
        line_start: Tuple[int, int],
        line_end: Tuple[int, int],
        target_classes: List[int] = None,
        bidirectional: bool = True
    ):
        """
        Initialize line crossing counter

        Args:
            line_start: 라인 시작점 (x, y)
            line_end: 라인 끝점 (x, y)
            target_classes: 카운팅할 클래스 ID
            bidirectional: 양방향 카운팅 여부
        """
        self.line_start = line_start
        self.line_end = line_end
        self.target_classes = target_classes
        self.bidirectional = bidirectional

        # 카운터
        self.count_in = 0  # 라인을 위->아래 또는 좌->우로 통과
        self.count_out = 0  # 라인을 아래->위 또는 우->좌로 통과

        # 객체 추적용
        self.tracked_objects: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.counted_objects = set()

        self.logger = logging.getLogger(__name__)

    def update(self, detections: List[dict]) -> Tuple[int, int, List[dict]]:
        """
        탐지 결과 업데이트 및 카운팅

        Args:
            detections: 탐지된 객체 리스트 (track_id 필수)

        Returns:
            (총 IN 카운트, 총 OUT 카운트, 방금 통과한 객체 리스트)
        """
        crossed_objects = []

        for det in detections:
            # track_id가 없으면 스킵
            if 'track_id' not in det or det['track_id'] is None:
                continue

            track_id = det['track_id']

            # 클래스 필터링
            if self.target_classes and det['class_id'] not in self.target_classes:
                continue

            center = det['center']

            # 이전 위치 가져오기
            prev_positions = self.tracked_objects[track_id]
            prev_positions.append(center)

            # 최근 10개 위치만 유지
            if len(prev_positions) > 10:
                prev_positions.pop(0)

            # 라인 통과 체크 (최소 2개 위치 필요)
            if len(prev_positions) >= 2 and track_id not in self.counted_objects:
                prev_center = prev_positions[-2]
                curr_center = prev_positions[-1]

                crossing = self._check_line_crossing(prev_center, curr_center)

                if crossing != 0:
                    # 통과 확인
                    self.counted_objects.add(track_id)

                    if crossing == 1:
                        self.count_in += 1
                        direction = "IN"
                    else:
                        self.count_out += 1
                        direction = "OUT"

                    crossed_objects.append({
                        **det,
                        'direction': direction
                    })

                    self.logger.info(
                        f"{det['class_name']} (ID: {track_id}) {direction} - "
                        f"IN: {self.count_in}, OUT: {self.count_out}"
                    )

        # 오래된 추적 정보 정리
        self._cleanup_old_tracks(detections)

        return self.count_in, self.count_out, crossed_objects

    def _check_line_crossing(
        self,
        prev_point: Tuple[int, int],
        curr_point: Tuple[int, int]
    ) -> int:
        """
        라인 통과 여부 확인

        Returns:
            1: IN (위->아래 또는 좌->우)
            -1: OUT (아래->위 또는 우->좌)
            0: 통과하지 않음
        """
        # 라인과의 교차 확인
        if not self._segments_intersect(
            prev_point, curr_point,
            self.line_start, self.line_end
        ):
            return 0

        # 방향 결정
        # 라인이 수평인 경우
        if abs(self.line_start[1] - self.line_end[1]) < abs(self.line_start[0] - self.line_end[0]):
            # 수평 라인: y 좌표로 방향 판단
            if prev_point[1] < curr_point[1]:
                return 1  # 위에서 아래로
            else:
                return -1 if self.bidirectional else 0
        else:
            # 수직 라인: x 좌표로 방향 판단
            if prev_point[0] < curr_point[0]:
                return 1  # 왼쪽에서 오른쪽으로
            else:
                return -1 if self.bidirectional else 0

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4) -> bool:
        """두 선분이 교차하는지 확인"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _cleanup_old_tracks(self, current_detections: List[dict]):
        """현재 탐지되지 않는 오래된 추적 정보 제거"""
        current_ids = {det.get('track_id') for det in current_detections if 'track_id' in det}

        # 더 이상 추적되지 않는 ID 제거
        old_ids = set(self.tracked_objects.keys()) - current_ids
        for old_id in old_ids:
            del self.tracked_objects[old_id]
            self.counted_objects.discard(old_id)

    def draw_line(
        self,
        frame: np.ndarray,
        color=(0, 255, 0),
        thickness=3
    ) -> np.ndarray:
        """
        카운팅 라인 그리기

        Args:
            frame: 입력 프레임
            color: 라인 색상
            thickness: 선 두께
        """
        cv2.line(
            frame,
            self.line_start,
            self.line_end,
            color,
            thickness
        )

        # 라인 중앙에 화살표 표시
        mid_x = (self.line_start[0] + self.line_end[0]) // 2
        mid_y = (self.line_start[1] + self.line_end[1]) // 2

        # 방향 화살표
        if abs(self.line_start[1] - self.line_end[1]) < abs(self.line_start[0] - self.line_end[0]):
            # 수평 라인
            cv2.arrowedLine(
                frame,
                (mid_x, mid_y - 20),
                (mid_x, mid_y + 20),
                (255, 255, 0),
                2
            )
        else:
            # 수직 라인
            cv2.arrowedLine(
                frame,
                (mid_x - 20, mid_y),
                (mid_x + 20, mid_y),
                (255, 255, 0),
                2
            )

        return frame

    def draw_counts(
        self,
        frame: np.ndarray,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        카운트 정보 표시

        Args:
            frame: 입력 프레임
            position: 텍스트 시작 위치
        """
        x, y = position

        # 배경 박스
        cv2.rectangle(
            frame,
            (x - 5, y - 25),
            (x + 200, y + 60),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            frame,
            (x - 5, y - 25),
            (x + 200, y + 60),
            (0, 255, 0),
            2
        )

        # 카운트 표시
        cv2.putText(
            frame,
            f"IN:  {self.count_in}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if self.bidirectional:
            cv2.putText(
                frame,
                f"OUT: {self.count_out}",
                (x, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            total = self.count_in + self.count_out
            cv2.putText(
                frame,
                f"Total: {total}",
                (x, y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return frame

    def reset(self):
        """카운터 리셋"""
        self.count_in = 0
        self.count_out = 0
        self.tracked_objects.clear()
        self.counted_objects.clear()
        self.logger.info("Counter reset")

    def get_statistics(self) -> dict:
        """통계 정보 반환"""
        return {
            'count_in': self.count_in,
            'count_out': self.count_out,
            'total': self.count_in + self.count_out,
            'currently_tracked': len(self.tracked_objects)
        }


if __name__ == "__main__":
    # 테스트 코드
    import sys
    sys.path.insert(0, '/home/user/rasp_vision_yolo')

    from src.camera import USBCamera
    from src.detection import YOLODetector

    logging.basicConfig(level=logging.INFO)

    # 카운팅 라인 설정 (화면 중앙 수평선)
    line_start = (0, 240)
    line_end = (640, 240)

    # 카운터 초기화 (사람만 카운팅)
    counter = LineCrossingCounter(
        line_start=line_start,
        line_end=line_end,
        target_classes=[0],  # 0 = person
        bidirectional=True
    )

    # 카메라 및 YOLO 초기화
    camera = USBCamera(width=640, height=480).start_threaded()
    yolo = YOLODetector(model_path='yolov8n.pt')

    print("라인 카운팅 시스템 시작")
    print("Press 'q' to quit, 'r' to reset")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            # 객체 탐지 및 추적
            results = yolo.model.track(frame, persist=True, verbose=False)

            # 추적 정보 포함한 탐지 결과 추출
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0]) if box.id is not None else None

                    det = {
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class_name': results[0].names[int(box.cls[0])],
                        'track_id': track_id
                    }
                    detections.append(det)

            # 카운팅 업데이트
            count_in, count_out, crossed = counter.update(detections)

            # 시각화
            frame = yolo.draw_detections(frame, results[0])
            frame = counter.draw_line(frame)
            frame = counter.draw_counts(frame)

            # 방금 통과한 객체 강조
            for obj in crossed:
                x1, y1, x2, y2 = obj['bbox']
                color = (0, 255, 0) if obj['direction'] == "IN" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            cv2.imshow('Line Crossing Counter', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.reset()

    finally:
        camera.stop()
        cv2.destroyAllWindows()
