"""
People Counting Example
라인 통과 카운팅 예제 - 리테일, 스마트시티 분야
"""

import cv2
import sys
import argparse
import logging
from pathlib import Path
import time

sys.path.insert(0, '/home/user/rasp_vision_yolo')

from src.camera import USBCamera, RTSPCamera, CSICamera
from src.detection import YOLODetector, LineCrossingCounter
from src.utils import TelegramNotifier


def main():
    parser = argparse.ArgumentParser(description='사람 카운팅 시스템')
    parser.add_argument('--camera', type=str, choices=['usb', 'csi', 'rtsp'], default='usb')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--url', type=str, help='RTSP URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.5)

    # 라인 설정
    parser.add_argument('--line', type=str, default='0,240,640,240',
                        help='카운팅 라인 (x1,y1,x2,y2)')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='양방향 카운팅')

    # 리포트 설정
    parser.add_argument('--report-interval', type=int, default=60,
                        help='리포트 전송 간격(초)')
    parser.add_argument('--telegram-token', type=str)
    parser.add_argument('--telegram-chat-id', type=str)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 라인 파싱
    line_coords = [int(x) for x in args.line.split(',')]
    line_start = (line_coords[0], line_coords[1])
    line_end = (line_coords[2], line_coords[3])

    # 카운터 초기화
    counter = LineCrossingCounter(
        line_start=line_start,
        line_end=line_end,
        target_classes=[0],  # person only
        bidirectional=args.bidirectional
    )

    # 알림 설정
    notifier = None
    if args.telegram_token and args.telegram_chat_id:
        notifier = TelegramNotifier(args.telegram_token, args.telegram_chat_id)
        print("✓ 텔레그램 리포트 활성화")

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
    print("사람 카운팅 시스템 시작")
    print("="*50)
    print("Controls:")
    print("  q: 종료")
    print("  r: 카운터 리셋")
    print("  s: 스크린샷 저장")
    print("="*50 + "\n")

    last_report_time = time.time()

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

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

            # 카운팅 업데이트
            count_in, count_out, crossed = counter.update(detections)

            # 시각화
            display_frame = frame.copy()
            display_frame = yolo.draw_detections(display_frame, results[0])
            display_frame = counter.draw_line(display_frame)
            display_frame = counter.draw_counts(display_frame)

            # 방금 통과한 객체 강조
            for obj in crossed:
                x1, y1, x2, y2 = obj['bbox']
                color = (0, 255, 0) if obj['direction'] == "IN" else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

                # 방향 텍스트
                cv2.putText(
                    display_frame,
                    f">>> {obj['direction']} >>>",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            # 정기 리포트 전송
            if notifier and (time.time() - last_report_time) >= args.report_interval:
                stats = counter.get_statistics()
                message = (
                    f"📊 카운팅 리포트\n\n"
                    f"IN: {stats['count_in']}명\n"
                    f"OUT: {stats['count_out']}명\n"
                    f"총계: {stats['total']}명\n"
                    f"현재 추적 중: {stats['currently_tracked']}명"
                )

                # 스크린샷 저장
                screenshot_dir = Path('recordings/reports')
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = screenshot_dir / f"report_{int(time.time())}.jpg"
                cv2.imwrite(str(screenshot_path), display_frame)

                notifier.send_alert('count', message, str(screenshot_path))
                last_report_time = time.time()
                print(f"\n📊 리포트 전송: {message}\n")

            cv2.imshow('People Counting System', display_frame)

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n시스템 종료")
                break
            elif key == ord('r'):
                counter.reset()
                print("\n카운터 리셋됨")
            elif key == ord('s'):
                filename = f"counting_{int(time.time())}.jpg"
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
        stats = counter.get_statistics()
        print("\n" + "="*50)
        print("최종 통계")
        print("="*50)
        print(f"IN: {stats['count_in']}명")
        print(f"OUT: {stats['count_out']}명")
        print(f"총 통과: {stats['total']}명")
        print("="*50)


if __name__ == "__main__":
    main()
