"""
Event-based Video Recorder
이벤트 기반 자동 녹화 시스템
특정 객체 탐지 시 자동으로 비디오 녹화
"""

import cv2
import sys
import argparse
import logging
from pathlib import Path
import time
from datetime import datetime
from collections import deque

sys.path.insert(0, '/home/user/rasp_vision_yolo')

from src.camera import USBCamera, RTSPCamera, CSICamera
from src.detection import YOLODetector
from src.utils import TelegramNotifier


class EventRecorder:
    """이벤트 기반 자동 녹화 시스템"""

    def __init__(
        self,
        output_dir: str = 'recordings/events',
        pre_event_seconds: int = 5,
        post_event_seconds: int = 10,
        fps: int = 20,
        max_file_size_mb: int = 500
    ):
        """
        Initialize event recorder

        Args:
            output_dir: 녹화 파일 저장 디렉토리
            pre_event_seconds: 이벤트 전 녹화 시간
            post_event_seconds: 이벤트 후 계속 녹화 시간
            fps: 녹화 FPS
            max_file_size_mb: 최대 파일 크기 (MB)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds
        self.fps = fps
        self.max_file_size_mb = max_file_size_mb

        # 프레임 버퍼 (이벤트 전 프레임 저장용)
        buffer_size = pre_event_seconds * fps
        self.frame_buffer = deque(maxlen=buffer_size)

        # 녹화 상태
        self.recording = False
        self.video_writer = None
        self.current_file = None
        self.event_start_time = None
        self.frames_after_event = 0
        self.post_event_frames = post_event_seconds * fps

        # 통계
        self.total_events = 0
        self.total_recordings = 0

        self.logger = logging.getLogger(__name__)

    def process_frame(
        self,
        frame,
        event_detected: bool,
        event_info: dict = None
    ):
        """
        프레임 처리 및 녹화

        Args:
            frame: 입력 프레임
            event_detected: 이벤트 발생 여부
            event_info: 이벤트 정보 (탐지된 객체 등)
        """
        # 항상 버퍼에 프레임 추가
        self.frame_buffer.append(frame.copy())

        if event_detected and not self.recording:
            # 새 이벤트 발생 - 녹화 시작
            self._start_recording(frame, event_info)

        elif self.recording:
            # 녹화 중
            self._write_frame(frame)

            if event_detected:
                # 이벤트 계속 발생 - 카운터 리셋
                self.frames_after_event = 0
            else:
                # 이벤트 종료 후
                self.frames_after_event += 1

                if self.frames_after_event >= self.post_event_frames:
                    # 충분히 녹화했으므로 종료
                    self._stop_recording()

            # 파일 크기 체크
            if self._check_file_size():
                self.logger.warning("최대 파일 크기 도달, 녹화 종료")
                self._stop_recording()

    def _start_recording(self, frame, event_info):
        """녹화 시작"""
        self.recording = True
        self.total_events += 1
        self.event_start_time = time.time()
        self.frames_after_event = 0

        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        event_type = event_info.get('type', 'event') if event_info else 'event'
        filename = f"{timestamp}_{event_type}.mp4"
        self.current_file = self.output_dir / filename

        # VideoWriter 초기화
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.current_file),
            fourcc,
            self.fps,
            (width, height)
        )

        # 버퍼에 있는 프레임들 먼저 쓰기 (이벤트 전 프레임)
        self.logger.info(f"녹화 시작: {filename}")
        for buffered_frame in self.frame_buffer:
            self.video_writer.write(buffered_frame)

        self.total_recordings += 1

    def _write_frame(self, frame):
        """프레임 쓰기"""
        if self.video_writer is not None:
            self.video_writer.write(frame)

    def _stop_recording(self):
        """녹화 종료"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

            duration = time.time() - self.event_start_time
            file_size = self.current_file.stat().st_size / (1024 * 1024)  # MB

            self.logger.info(
                f"녹화 완료: {self.current_file.name} "
                f"({duration:.1f}초, {file_size:.1f}MB)"
            )

            self.recording = False
            self.current_file = None

    def _check_file_size(self) -> bool:
        """파일 크기 체크"""
        if self.current_file and self.current_file.exists():
            size_mb = self.current_file.stat().st_size / (1024 * 1024)
            return size_mb >= self.max_file_size_mb
        return False

    def cleanup(self):
        """정리"""
        if self.recording:
            self._stop_recording()

    def get_statistics(self) -> dict:
        """통계 정보"""
        return {
            'total_events': self.total_events,
            'total_recordings': self.total_recordings,
            'currently_recording': self.recording
        }


def main():
    parser = argparse.ArgumentParser(description='이벤트 기반 자동 녹화 시스템')
    parser.add_argument('--camera', type=str, choices=['usb', 'csi', 'rtsp'], default='usb')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--url', type=str, help='RTSP URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.5)

    # 녹화 설정
    parser.add_argument('--output-dir', type=str, default='recordings/events',
                        help='녹화 파일 저장 디렉토리')
    parser.add_argument('--pre-event', type=int, default=5,
                        help='이벤트 전 녹화 시간(초)')
    parser.add_argument('--post-event', type=int, default=10,
                        help='이벤트 후 녹화 시간(초)')
    parser.add_argument('--fps', type=int, default=20,
                        help='녹화 FPS')

    # 이벤트 트리거
    parser.add_argument('--trigger-classes', type=int, nargs='+', default=[0],
                        help='녹화를 트리거할 클래스 ID')
    parser.add_argument('--min-objects', type=int, default=1,
                        help='녹화를 시작할 최소 객체 수')

    # 알림
    parser.add_argument('--telegram-token', type=str)
    parser.add_argument('--telegram-chat-id', type=str)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 녹화기 초기화
    recorder = EventRecorder(
        output_dir=args.output_dir,
        pre_event_seconds=args.pre_event,
        post_event_seconds=args.post_event,
        fps=args.fps
    )

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
        classes=args.trigger_classes
    )

    print("\n" + "="*50)
    print("이벤트 기반 자동 녹화 시스템 시작")
    print("="*50)
    print(f"트리거 클래스: {args.trigger_classes}")
    print(f"최소 객체 수: {args.min_objects}")
    print(f"저장 위치: {args.output_dir}")
    print("Controls:")
    print("  q: 종료")
    print("  r: 수동 녹화 시작/중지")
    print("="*50 + "\n")

    manual_recording = False
    last_notification = 0

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            # YOLO 탐지
            results = yolo.detect(frame)
            detections = yolo.get_detection_info(results)

            # 이벤트 감지
            event_detected = len(detections) >= args.min_objects or manual_recording

            # 이벤트 정보
            event_info = None
            if event_detected and detections:
                class_names = [d['class_name'] for d in detections]
                event_info = {
                    'type': '_'.join(set(class_names)),
                    'count': len(detections)
                }

            # 녹화 처리
            recorder.process_frame(frame, event_detected, event_info)

            # 알림 (녹화 시작 시 한 번만)
            if event_detected and not recorder.recording and notifier:
                current_time = time.time()
                if current_time - last_notification > 60:  # 1분에 한 번
                    message = (
                        f"🎥 녹화 시작\n\n"
                        f"탐지: {len(detections)}개 객체\n"
                        f"클래스: {', '.join(set([d['class_name'] for d in detections]))}"
                    )
                    notifier.send_message(message)
                    last_notification = current_time

            # 시각화
            display_frame = frame.copy()
            display_frame = yolo.draw_detections(display_frame, results)

            # 녹화 상태 표시
            status_color = (0, 0, 255) if recorder.recording else (0, 255, 0)
            status_text = "● REC" if recorder.recording else "○ STANDBY"

            cv2.putText(
                display_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_BOLD,
                1,
                status_color,
                2
            )

            # 통계
            stats = recorder.get_statistics()
            cv2.putText(
                display_frame,
                f"Events: {stats['total_events']} | Files: {stats['total_recordings']}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow('Event Recorder', display_frame)

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n시스템 종료")
                break
            elif key == ord('r'):
                manual_recording = not manual_recording
                print(f"\n수동 녹화: {'ON' if manual_recording else 'OFF'}")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단됨")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        recorder.cleanup()
        camera.stop()
        cv2.destroyAllWindows()

        # 최종 통계
        stats = recorder.get_statistics()
        print("\n" + "="*50)
        print("최종 통계")
        print("="*50)
        print(f"총 이벤트: {stats['total_events']}")
        print(f"총 녹화 파일: {stats['total_recordings']}")
        print(f"저장 위치: {args.output_dir}")
        print("="*50)


if __name__ == "__main__":
    main()
