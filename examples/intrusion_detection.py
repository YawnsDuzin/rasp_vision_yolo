"""
Intrusion Detection Example
침입 감지 시스템 예제 - 보안 및 감시 분야
"""

import cv2
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, '/home/user/rasp_vision_yolo')

from src.camera import USBCamera, RTSPCamera, CSICamera
from src.detection import YOLODetector, IntrusionDetector
from src.utils import TelegramNotifier


def main():
    parser = argparse.ArgumentParser(description='침입 감지 시스템')
    parser.add_argument('--camera', type=str, choices=['usb', 'csi', 'rtsp'], default='usb')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--url', type=str, help='RTSP URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.5)

    # ROI 설정
    parser.add_argument('--roi', type=str, default='200,150,440,150,440,330,200,330',
                        help='ROI 좌표 (x1,y1,x2,y2,...)')

    # 알림 설정
    parser.add_argument('--telegram-token', type=str, help='텔레그램 봇 토큰')
    parser.add_argument('--telegram-chat-id', type=str, help='텔레그램 채팅 ID')
    parser.add_argument('--save-alerts', action='store_true', help='알림 이미지 저장')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # ROI 파싱
    roi_coords = [int(x) for x in args.roi.split(',')]
    roi_points = [(roi_coords[i], roi_coords[i+1]) for i in range(0, len(roi_coords), 2)]

    # 침입 감지기 초기화
    intrusion_detector = IntrusionDetector(
        roi_points=roi_points,
        target_classes=[0],  # 0 = person
        cooldown_seconds=5,
        min_confidence=args.conf
    )

    # 알림 설정
    notifier = None
    if args.telegram_token and args.telegram_chat_id:
        notifier = TelegramNotifier(args.telegram_token, args.telegram_chat_id)
        print("✓ 텔레그램 알림 활성화")

    # 저장 디렉토리
    if args.save_alerts:
        alert_dir = Path('recordings/alerts')
        alert_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ 알림 이미지 저장: {alert_dir}")

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
            print("Error: RTSP URL이 필요합니다 (--url)")
            return
        camera = RTSPCamera(rtsp_url=args.url).start_threaded()

    # YOLO 초기화
    print(f"YOLO 모델 로드: {args.model}")
    yolo = YOLODetector(model_path=args.model, conf_threshold=args.conf)

    print("\n" + "="*50)
    print("침입 감지 시스템 시작")
    print("="*50)
    print("Controls:")
    print("  q: 종료")
    print("  r: 통계 초기화")
    print("  s: 스크린샷 저장")
    print("="*50 + "\n")

    try:
        frame_count = 0

        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            frame_count += 1

            # YOLO 탐지
            results = yolo.detect(frame)
            detections = yolo.get_detection_info(results)

            # 침입 체크
            intrusion, intruders = intrusion_detector.check_intrusion(detections)

            # 시각화
            display_frame = frame.copy()
            display_frame = intrusion_detector.draw_roi(display_frame)
            display_frame = yolo.draw_detections(display_frame, results)

            if intrusion:
                display_frame = intrusion_detector.draw_intruders(display_frame, intruders)

                # 알림 발송
                if intrusion_detector.should_alert():
                    intrusion_detector.trigger_alert()

                    message = (
                        f"⚠️ 침입 감지!\n\n"
                        f"위치: 지정된 보안 구역\n"
                        f"침입자 수: {len(intruders)}명\n"
                        f"총 침입 횟수: {intrusion_detector.intrusion_count}회"
                    )

                    print(f"\n🚨 {message}\n")

                    # 이미지 저장
                    if args.save_alerts:
                        import time
                        filename = alert_dir / f"alert_{int(time.time())}.jpg"
                        cv2.imwrite(str(filename), display_frame)

                        # 텔레그램 전송
                        if notifier:
                            notifier.send_alert('intrusion', message, str(filename))
                    elif notifier:
                        # 이미지 없이 메시지만 전송
                        notifier.send_alert('intrusion', message)

            # 통계 표시
            stats = intrusion_detector.get_statistics()
            y_pos = 30
            for key, value in stats.items():
                text = f"{key}: {value}"
                cv2.putText(
                    display_frame,
                    text,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                y_pos += 30

            # 표시
            cv2.imshow('Intrusion Detection System', display_frame)

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n시스템 종료")
                break
            elif key == ord('r'):
                intrusion_detector.reset_statistics()
                print("\n통계 초기화됨")
            elif key == ord('s'):
                import time
                filename = f"screenshot_{int(time.time())}.jpg"
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
        stats = intrusion_detector.get_statistics()
        print("\n" + "="*50)
        print("최종 통계")
        print("="*50)
        print(f"총 침입 횟수: {stats['total_intrusions']}")
        print(f"처리된 프레임: {frame_count}")
        print("="*50)


if __name__ == "__main__":
    main()
