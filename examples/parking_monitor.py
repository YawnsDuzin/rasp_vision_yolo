"""
Parking Space Monitoring Example
주차 공간 모니터링 예제 - 스마트시티 분야
"""

import cv2
import sys
import argparse
import logging
import yaml
from pathlib import Path

sys.path.insert(0, '/home/user/rasp_vision_yolo')

from src.camera import USBCamera, RTSPCamera, CSICamera
from src.detection import YOLODetector, ParkingSpaceMonitor


def load_parking_config(config_path):
    """주차 공간 설정 로드"""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('parking_spaces', [])
    return None


def create_default_config():
    """기본 주차 공간 설정 (4개 공간)"""
    return [
        {
            'id': 1,
            'points': [(50, 200), (150, 200), (150, 350), (50, 350)],
            'reserved': False
        },
        {
            'id': 2,
            'points': [(170, 200), (270, 200), (270, 350), (170, 350)],
            'reserved': False
        },
        {
            'id': 3,
            'points': [(290, 200), (390, 200), (390, 350), (290, 350)],
            'reserved': False
        },
        {
            'id': 4,
            'points': [(410, 200), (510, 200), (510, 350), (410, 350)],
            'reserved': True  # 예약석
        }
    ]


def main():
    parser = argparse.ArgumentParser(description='주차 공간 모니터링 시스템')
    parser.add_argument('--camera', type=str, choices=['usb', 'csi', 'rtsp'], default='usb')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--url', type=str, help='RTSP URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--config', type=str, help='주차 공간 설정 파일 (YAML)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 주차 공간 설정 로드
    parking_spaces = None
    if args.config:
        parking_spaces = load_parking_config(args.config)

    if parking_spaces is None:
        print("기본 주차 공간 설정 사용 (4개 공간)")
        parking_spaces = create_default_config()

    # 주차 모니터 초기화
    monitor = ParkingSpaceMonitor(parking_spaces)

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
        classes=[2, 3, 5, 7]  # car, motorcycle, bus, truck
    )

    print("\n" + "="*50)
    print("주차 공간 모니터링 시스템 시작")
    print("="*50)
    print(f"총 주차 공간: {len(parking_spaces)}개")
    print("Controls:")
    print("  q: 종료")
    print("  s: 스크린샷 저장")
    print("="*50 + "\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            # YOLO 차량 탐지
            results = yolo.detect(frame)
            vehicles = yolo.get_detection_info(results)

            # 주차 공간 점유 상태 확인
            space_status = monitor.check_occupancy(vehicles)

            # 시각화
            display_frame = frame.copy()

            # 주차 공간 그리기
            display_frame = monitor.draw_spaces(display_frame, space_status)

            # 차량 박스 그리기
            for vehicle in vehicles:
                x1, y1, x2, y2 = vehicle['bbox']
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{vehicle['class_name']} {vehicle['confidence']:.2f}"
                cv2.putText(
                    display_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

            # 통계 표시
            stats = monitor.get_statistics()

            # 통계 박스 배경
            cv2.rectangle(
                display_frame,
                (10, 10),
                (250, 140),
                (0, 0, 0),
                -1
            )
            cv2.rectangle(
                display_frame,
                (10, 10),
                (250, 140),
                (255, 255, 255),
                2
            )

            # 통계 텍스트
            stats_text = [
                f"Total Spaces: {stats['total']}",
                f"Occupied: {stats['occupied']}",
                f"Available: {stats['available']}",
                f"Occupancy: {stats['occupancy_rate']:.1f}%"
            ]

            y_pos = 35
            for text in stats_text:
                cv2.putText(
                    display_frame,
                    text,
                    (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                y_pos += 30

            # 표시
            cv2.imshow('Parking Space Monitor', display_frame)

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n시스템 종료")
                break
            elif key == ord('s'):
                import time
                filename = f"parking_{int(time.time())}.jpg"
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
        stats = monitor.get_statistics()
        print("\n" + "="*50)
        print("최종 통계")
        print("="*50)
        print(f"총 주차 공간: {stats['total']}개")
        print(f"점유: {stats['occupied']}개")
        print(f"가용: {stats['available']}개")
        print(f"점유율: {stats['occupancy_rate']:.1f}%")
        print("="*50)


if __name__ == "__main__":
    main()
