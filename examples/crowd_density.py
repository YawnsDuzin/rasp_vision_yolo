"""
Crowd Density Monitoring
혼잡도 분석 시스템 - 스마트시티, 리테일, 이벤트 관리
"""

import cv2
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
import time
from collections import deque

sys.path.insert(0, '/home/user/rasp_vision_yolo')

from src.camera import USBCamera, RTSPCamera, CSICamera
from src.detection import YOLODetector
from src.utils import TelegramNotifier


class CrowdDensityMonitor:
    """혼잡도 모니터링 시스템"""

    def __init__(
        self,
        zones: dict,
        density_threshold: dict = None,
        history_size: int = 30
    ):
        """
        Initialize crowd density monitor

        Args:
            zones: 모니터링 영역 {'zone_name': [(x1,y1), (x2,y2), ...]}
            density_threshold: 영역별 혼잡도 임계값 {'low': 5, 'medium': 10, 'high': 15}
            history_size: 히스토리 저장 크기
        """
        self.zones = zones
        self.density_threshold = density_threshold or {
            'low': 5,
            'medium': 10,
            'high': 15
        }

        # 각 영역별 히스토리
        self.zone_history = {
            name: deque(maxlen=history_size)
            for name in zones.keys()
        }

        # 영역별 색상
        self.zone_colors = self._generate_zone_colors()

        self.logger = logging.getLogger(__name__)

    def _generate_zone_colors(self):
        """영역별 고유 색상 생성"""
        colors = {}
        for i, name in enumerate(self.zones.keys()):
            np.random.seed(hash(name) % 2**32)
            colors[name] = tuple(np.random.randint(0, 255, 3).tolist())
        return colors

    def analyze_density(self, detections: list) -> dict:
        """
        각 영역의 혼잡도 분석

        Args:
            detections: 탐지된 객체 리스트

        Returns:
            영역별 혼잡도 정보
        """
        zone_counts = {name: 0 for name in self.zones.keys()}
        zone_objects = {name: [] for name in self.zones.keys()}

        # 각 탐지 객체가 어느 영역에 속하는지 확인
        for det in detections:
            center = det['center']

            for zone_name, zone_points in self.zones.items():
                polygon = np.array(zone_points, dtype=np.int32)
                if cv2.pointPolygonTest(polygon, center, False) >= 0:
                    zone_counts[zone_name] += 1
                    zone_objects[zone_name].append(det)
                    break  # 한 영역에만 속하도록

        # 혼잡도 레벨 결정
        zone_density = {}
        for zone_name, count in zone_counts.items():
            level = self._get_density_level(count)
            zone_density[zone_name] = {
                'count': count,
                'level': level,
                'objects': zone_objects[zone_name]
            }

            # 히스토리 저장
            self.zone_history[zone_name].append(count)

        return zone_density

    def _get_density_level(self, count: int) -> str:
        """카운트를 기반으로 혼잡도 레벨 결정"""
        if count >= self.density_threshold['high']:
            return 'HIGH'
        elif count >= self.density_threshold['medium']:
            return 'MEDIUM'
        elif count >= self.density_threshold['low']:
            return 'LOW'
        else:
            return 'EMPTY'

    def draw_zones(self, frame: np.ndarray, zone_density: dict) -> np.ndarray:
        """영역 및 혼잡도 시각화"""
        overlay = frame.copy()

        for zone_name, density_info in zone_density.items():
            points = np.array(self.zones[zone_name], dtype=np.int32)
            level = density_info['level']
            count = density_info['count']

            # 레벨별 색상
            if level == 'HIGH':
                color = (0, 0, 255)  # 빨강
            elif level == 'MEDIUM':
                color = (0, 165, 255)  # 주황
            elif level == 'LOW':
                color = (0, 255, 255)  # 노랑
            else:
                color = (0, 255, 0)  # 녹색

            # 영역 그리기
            cv2.polylines(frame, [points], True, color, 3)
            cv2.fillPoly(overlay, [points], color)

            # 중심점 계산
            M = cv2.moments(points)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 텍스트 배경
                text = f"{zone_name}"
                text2 = f"{level}: {count}"

                (w1, h1), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_BOLD, 0.8, 2)
                (w2, h2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                max_w = max(w1, w2)
                total_h = h1 + h2 + 20

                cv2.rectangle(
                    frame,
                    (cx - max_w // 2 - 10, cy - total_h // 2),
                    (cx + max_w // 2 + 10, cy + total_h // 2),
                    (0, 0, 0),
                    -1
                )

                # 텍스트
                cv2.putText(
                    frame,
                    text,
                    (cx - w1 // 2, cy - h2 // 2 - 5),
                    cv2.FONT_HERSHEY_BOLD,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    text2,
                    (cx - w2 // 2, cy + h1 // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # 오버레이 적용
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        return frame

    def draw_statistics(self, frame: np.ndarray, zone_density: dict) -> np.ndarray:
        """통계 정보 표시"""
        y_offset = 30
        x_offset = 10

        # 배경
        total_zones = len(self.zones)
        bg_height = 40 + total_zones * 30
        cv2.rectangle(
            frame,
            (x_offset - 5, y_offset - 25),
            (x_offset + 250, y_offset + bg_height),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            frame,
            (x_offset - 5, y_offset - 25),
            (x_offset + 250, y_offset + bg_height),
            (255, 255, 255),
            2
        )

        # 제목
        cv2.putText(
            frame,
            "Crowd Density Monitor",
            (x_offset, y_offset),
            cv2.FONT_HERSHEY_BOLD,
            0.6,
            (255, 255, 255),
            2
        )

        y_offset += 35

        # 각 영역 정보
        for zone_name, density_info in zone_density.items():
            level = density_info['level']
            count = density_info['count']

            # 평균 계산
            history = list(self.zone_history[zone_name])
            avg_count = np.mean(history) if history else 0

            # 색상
            if level == 'HIGH':
                color = (0, 0, 255)
            elif level == 'MEDIUM':
                color = (0, 165, 255)
            elif level == 'LOW':
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            text = f"{zone_name}: {count} (avg: {avg_count:.1f})"

            cv2.putText(
                frame,
                text,
                (x_offset, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

            y_offset += 30

        return frame

    def get_alert_zones(self, zone_density: dict) -> list:
        """경고가 필요한 영역 반환"""
        alerts = []

        for zone_name, density_info in zone_density.items():
            if density_info['level'] in ['HIGH', 'MEDIUM']:
                alerts.append({
                    'zone': zone_name,
                    'level': density_info['level'],
                    'count': density_info['count']
                })

        return alerts


def main():
    parser = argparse.ArgumentParser(description='혼잡도 모니터링 시스템')
    parser.add_argument('--camera', type=str, choices=['usb', 'csi', 'rtsp'], default='usb')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--url', type=str, help='RTSP URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.5)

    # 혼잡도 임계값
    parser.add_argument('--threshold-low', type=int, default=5)
    parser.add_argument('--threshold-medium', type=int, default=10)
    parser.add_argument('--threshold-high', type=int, default=15)

    # 알림
    parser.add_argument('--telegram-token', type=str)
    parser.add_argument('--telegram-chat-id', type=str)
    parser.add_argument('--alert-interval', type=int, default=300,
                        help='알림 간격 (초)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 모니터링 영역 정의 (3개 영역 예시)
    zones = {
        'Zone A': [(50, 100), (250, 100), (250, 350), (50, 350)],
        'Zone B': [(270, 100), (470, 100), (470, 350), (270, 350)],
        'Zone C': [(490, 100), (630, 100), (630, 350), (490, 350)]
    }

    # 혼잡도 모니터 초기화
    density_threshold = {
        'low': args.threshold_low,
        'medium': args.threshold_medium,
        'high': args.threshold_high
    }

    monitor = CrowdDensityMonitor(zones, density_threshold)

    # 알림 설정
    notifier = None
    if args.telegram_token and args.telegram_chat_id:
        notifier = TelegramNotifier(args.telegram_token, args.telegram_chat_id)
        print("✓ 텔레그램 알림 활성화")

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
    yolo = YOLODetector(
        model_path=args.model,
        conf_threshold=args.conf,
        classes=[0]  # person only
    )

    print("\n" + "="*50)
    print("혼잡도 모니터링 시스템 시작")
    print("="*50)
    print("Controls:")
    print("  q: 종료")
    print("  s: 스크린샷 저장")
    print("="*50 + "\n")

    last_alert_time = time.time()

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            # YOLO 탐지
            results = yolo.detect(frame)
            detections = yolo.get_detection_info(results)

            # 혼잡도 분석
            zone_density = monitor.analyze_density(detections)

            # 시각화
            display_frame = frame.copy()
            display_frame = monitor.draw_zones(display_frame, zone_density)
            display_frame = yolo.draw_detections(display_frame, results)
            display_frame = monitor.draw_statistics(display_frame, zone_density)

            # 경고 체크
            alerts = monitor.get_alert_zones(zone_density)

            if alerts and notifier:
                elapsed = time.time() - last_alert_time
                if elapsed >= args.alert_interval:
                    # 알림 메시지 생성
                    message_lines = ["🚨 혼잡도 경고!", ""]
                    for alert in alerts:
                        message_lines.append(
                            f"📍 {alert['zone']}: {alert['level']} ({alert['count']}명)"
                        )

                    message = "\n".join(message_lines)

                    # 스크린샷 저장
                    screenshot_dir = Path('recordings/crowd_alerts')
                    screenshot_dir.mkdir(parents=True, exist_ok=True)
                    screenshot_path = screenshot_dir / f"alert_{int(time.time())}.jpg"
                    cv2.imwrite(str(screenshot_path), display_frame)

                    # 알림 전송
                    notifier.send_alert('alert', message, str(screenshot_path))
                    last_alert_time = time.time()
                    print(f"\n{message}\n")

            cv2.imshow('Crowd Density Monitor', display_frame)

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n시스템 종료")
                break
            elif key == ord('s'):
                filename = f"crowd_{int(time.time())}.jpg"
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


if __name__ == "__main__":
    main()
