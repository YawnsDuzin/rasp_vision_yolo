"""
Advanced Object Tracker
고급 객체 추적 - 체류 시간, 이동 경로, 속도 분석
"""

import cv2
import sys
import argparse
import logging
from pathlib import Path
import time
import numpy as np
from collections import defaultdict, deque
from datetime import datetime

sys.path.insert(0, '/home/user/rasp_vision_yolo')

from src.camera import USBCamera, RTSPCamera, CSICamera
from src.detection import YOLODetector


class AdvancedObjectTracker:
    """고급 객체 추적 시스템"""

    def __init__(
        self,
        track_history_size: int = 100,
        dwell_threshold_seconds: int = 3,
        speed_calculation_window: int = 10
    ):
        """
        Initialize advanced tracker

        Args:
            track_history_size: 추적 히스토리 크기
            dwell_threshold_seconds: 체류 시간 임계값
            speed_calculation_window: 속도 계산 윈도우
        """
        self.track_history_size = track_history_size
        self.dwell_threshold_seconds = dwell_threshold_seconds
        self.speed_calculation_window = speed_calculation_window

        # 추적 데이터
        self.tracks = {}  # {track_id: TrackInfo}
        self.track_history = defaultdict(lambda: deque(maxlen=track_history_size))

        # 통계
        self.total_tracked = 0
        self.active_tracks = 0

        self.logger = logging.getLogger(__name__)

    def update(self, detections: list, timestamp: float = None):
        """
        추적 정보 업데이트

        Args:
            detections: 탐지 결과 (track_id 포함)
            timestamp: 현재 타임스탬프
        """
        if timestamp is None:
            timestamp = time.time()

        current_track_ids = set()

        for det in detections:
            if 'track_id' not in det or det['track_id'] is None:
                continue

            track_id = det['track_id']
            current_track_ids.add(track_id)

            # 새로운 트랙
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'positions': deque(maxlen=self.speed_calculation_window),
                    'timestamps': deque(maxlen=self.speed_calculation_window),
                    'class_name': det['class_name'],
                    'total_distance': 0,
                    'avg_speed': 0
                }
                self.total_tracked += 1

            # 트랙 정보 업데이트
            track = self.tracks[track_id]
            track['last_seen'] = timestamp

            center = det['center']

            # 이전 위치가 있으면 거리 계산
            if track['positions']:
                prev_center = track['positions'][-1]
                distance = np.sqrt(
                    (center[0] - prev_center[0])**2 +
                    (center[1] - prev_center[1])**2
                )
                track['total_distance'] += distance

            track['positions'].append(center)
            track['timestamps'].append(timestamp)

            # 속도 계산 (픽셀/초)
            if len(track['positions']) >= 2:
                speed = self._calculate_speed(track)
                track['avg_speed'] = speed

            # 히스토리 저장
            self.track_history[track_id].append({
                'timestamp': timestamp,
                'center': center,
                'bbox': det['bbox']
            })

        # 사라진 트랙 정리
        disappeared_tracks = set(self.tracks.keys()) - current_track_ids
        for track_id in disappeared_tracks:
            if timestamp - self.tracks[track_id]['last_seen'] > 5:
                del self.tracks[track_id]
                if track_id in self.track_history:
                    del self.track_history[track_id]

        self.active_tracks = len(current_track_ids)

    def _calculate_speed(self, track: dict) -> float:
        """속도 계산 (픽셀/초)"""
        if len(track['positions']) < 2:
            return 0

        positions = list(track['positions'])
        timestamps = list(track['timestamps'])

        # 총 이동 거리
        total_distance = 0
        for i in range(1, len(positions)):
            dist = np.sqrt(
                (positions[i][0] - positions[i-1][0])**2 +
                (positions[i][1] - positions[i-1][1])**2
            )
            total_distance += dist

        # 총 시간
        total_time = timestamps[-1] - timestamps[0]

        if total_time > 0:
            return total_distance / total_time
        return 0

    def get_dwell_time(self, track_id: int) -> float:
        """체류 시간 계산 (초)"""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            return track['last_seen'] - track['first_seen']
        return 0

    def get_long_dwell_objects(self) -> list:
        """오래 체류한 객체 반환"""
        long_dwell = []

        for track_id, track in self.tracks.items():
            dwell_time = track['last_seen'] - track['first_seen']
            if dwell_time >= self.dwell_threshold_seconds:
                long_dwell.append({
                    'track_id': track_id,
                    'dwell_time': dwell_time,
                    'class_name': track['class_name'],
                    'last_position': track['positions'][-1] if track['positions'] else None
                })

        return long_dwell

    def draw_tracks(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """추적 정보 시각화"""
        for det in detections:
            if 'track_id' not in det or det['track_id'] is None:
                continue

            track_id = det['track_id']

            if track_id not in self.track_history:
                continue

            # 이동 경로 그리기
            history = list(self.track_history[track_id])
            if len(history) > 1:
                points = np.array([h['center'] for h in history], dtype=np.int32)
                cv2.polylines(frame, [points], False, (0, 255, 0), 2)

            # 현재 위치
            x1, y1, x2, y2 = det['bbox']
            center = det['center']

            # 트랙 정보
            if track_id in self.tracks:
                track = self.tracks[track_id]
                dwell_time = self.get_dwell_time(track_id)
                speed = track['avg_speed']

                # ID 및 정보 표시
                info_text = f"ID:{track_id}"
                cv2.putText(
                    frame,
                    info_text,
                    (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # 체류 시간
                dwell_text = f"Dwell:{dwell_time:.1f}s"
                cv2.putText(
                    frame,
                    dwell_text,
                    (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )

                # 속도
                speed_text = f"Speed:{speed:.1f}px/s"
                cv2.putText(
                    frame,
                    speed_text,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1
                )

                # 오래 체류한 객체 강조
                if dwell_time >= self.dwell_threshold_seconds:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        return frame

    def draw_heatmap(self, frame_shape: tuple, alpha: float = 0.5) -> np.ndarray:
        """히트맵 생성 (체류 위치 시각화)"""
        heatmap = np.zeros(frame_shape[:2], dtype=np.float32)

        for track_id, history in self.track_history.items():
            for record in history:
                center = record['center']
                # 가우시안 분포로 히트맵 생성
                cv2.circle(heatmap, center, 20, 1, -1)

        # 정규화
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # 컬러맵 적용
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        return heatmap_colored

    def get_statistics(self) -> dict:
        """통계 정보"""
        return {
            'total_tracked': self.total_tracked,
            'active_tracks': self.active_tracks,
            'long_dwell_count': len(self.get_long_dwell_objects())
        }


def main():
    parser = argparse.ArgumentParser(description='고급 객체 추적 시스템')
    parser.add_argument('--camera', type=str, choices=['usb', 'csi', 'rtsp'], default='usb')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--url', type=str, help='RTSP URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.5)

    # 추적 설정
    parser.add_argument('--dwell-threshold', type=int, default=3,
                        help='체류 시간 임계값(초)')
    parser.add_argument('--show-heatmap', action='store_true',
                        help='히트맵 표시')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 추적기 초기화
    tracker = AdvancedObjectTracker(
        dwell_threshold_seconds=args.dwell_threshold
    )

    # 카메라 초기화
    print(f"카메라 초기화: {args.camera.upper()}")

    if args.camera == 'usb':
        camera = USBCamera(device_id=args.device).start_threaded()
    elif args.camera == 'csi':
        if CSICamera is None:
            print("Error: CSI 카메라를 사용할 수 없습니다")
            return
        camera = CSICamera().start()
    elif args.camera == 'rtsp':
        if not args.url:
            print("Error: RTSP URL이 필요합니다")
            return
        camera = RTSPCamera(rtsp_url=args.url).start_threaded()

    # YOLO 초기화
    print(f"YOLO 모델 로드: {args.model}")
    yolo = YOLODetector(model_path=args.model, conf_threshold=args.conf)

    print("\n" + "="*50)
    print("고급 객체 추적 시스템 시작")
    print("="*50)
    print("Controls:")
    print("  q: 종료")
    print("  h: 히트맵 토글")
    print("  s: 스크린샷 저장")
    print("="*50 + "\n")

    show_heatmap = args.show_heatmap

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            timestamp = time.time()

            # YOLO 탐지 + 추적
            results = yolo.model.track(frame, persist=True, verbose=False, conf=args.conf)

            # 탐지 결과 파싱
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

            # 추적 업데이트
            tracker.update(detections, timestamp)

            # 시각화
            display_frame = frame.copy()

            if show_heatmap:
                # 히트맵 오버레이
                heatmap = tracker.draw_heatmap(frame.shape)
                display_frame = cv2.addWeighted(display_frame, 0.7, heatmap, 0.3, 0)

            display_frame = tracker.draw_tracks(display_frame, detections)

            # 오래 체류한 객체 알림
            long_dwell = tracker.get_long_dwell_objects()
            if long_dwell:
                y_offset = 30
                for obj in long_dwell:
                    text = f"⚠ ID {obj['track_id']}: {obj['dwell_time']:.1f}s"
                    cv2.putText(
                        display_frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
                    y_offset += 30

            # 통계
            stats = tracker.get_statistics()
            cv2.putText(
                display_frame,
                f"Active: {stats['active_tracks']} | Total: {stats['total_tracked']}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow('Advanced Object Tracker', display_frame)

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n시스템 종료")
                break
            elif key == ord('h'):
                show_heatmap = not show_heatmap
                print(f"\n히트맵: {'ON' if show_heatmap else 'OFF'}")
            elif key == ord('s'):
                filename = f"tracker_{int(time.time())}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"\n스크린샷 저장: {filename}")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단됨")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        camera.stop()
        cv2.destroyAllWindows()

        # 최종 통계
        stats = tracker.get_statistics()
        print("\n" + "="*50)
        print("최종 통계")
        print("="*50)
        print(f"총 추적 객체: {stats['total_tracked']}")
        print(f"활성 추적: {stats['active_tracks']}")
        print("="*50)


if __name__ == "__main__":
    main()
