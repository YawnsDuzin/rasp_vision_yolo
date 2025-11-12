"""
Fall Detection System using YOLOv8-pose
포즈 추정 기반 낙상 감지 시스템
의료, 헬스케어, 요양원 등에 활용
"""

import cv2
import sys
import argparse
import logging
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, '/home/user/rasp_vision_yolo')

from src.camera import USBCamera, RTSPCamera, CSICamera
from src.detection.pose_analyzer import PoseAnalyzer
from src.utils import TelegramNotifier
from ultralytics import YOLO


class FallDetectionSystem:
    """낙상 감지 시스템"""

    def __init__(
        self,
        model_path: str = 'yolov8n-pose.pt',
        pose_analyzer: PoseAnalyzer = None,
        notifier: TelegramNotifier = None,
        record_falls: bool = True,
        output_dir: str = 'recordings/falls'
    ):
        """
        Initialize fall detection system

        Args:
            model_path: YOLOv8-pose 모델 경로
            pose_analyzer: 포즈 분석기
            notifier: 알림 시스템
            record_falls: 낙상 시 녹화 여부
            output_dir: 녹화 파일 저장 디렉토리
        """
        self.model = YOLO(model_path)
        self.pose_analyzer = pose_analyzer or PoseAnalyzer()
        self.notifier = notifier
        self.record_falls = record_falls

        if record_falls:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.video_writer = None
        self.recording = False
        self.last_alert_time = 0
        self.alert_cooldown = 60  # 60초 쿨다운

        self.logger = logging.getLogger(__name__)

    def process_frame(self, frame: np.ndarray, timestamp: float = None):
        """
        프레임 처리 및 낙상 감지

        Args:
            frame: 입력 프레임
            timestamp: 타임스탬프

        Returns:
            (annotated_frame, fall_detected, analysis_results)
        """
        if timestamp is None:
            timestamp = time.time()

        # 포즈 추정
        results = self.model(frame, verbose=False)

        annotated_frame = frame.copy()
        fall_detected = False
        all_analysis = []

        # 각 사람에 대해 포즈 분석
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints

            for idx in range(len(keypoints_data)):
                # 키포인트 추출
                kp = keypoints_data[idx]

                if kp.xy is not None and len(kp.xy) > 0:
                    keypoints = kp.xy[0].cpu().numpy()  # (17, 2)
                    keypoint_conf = kp.conf[0].cpu().numpy()  # (17,)

                    # 포즈 분석
                    analysis = self.pose_analyzer.analyze_pose(keypoints, keypoint_conf)
                    all_analysis.append(analysis)

                    # 시각화
                    annotated_frame = self._draw_pose(
                        annotated_frame,
                        keypoints,
                        keypoint_conf,
                        analysis
                    )

                    # 낙상 감지
                    if analysis['is_fall']:
                        fall_detected = True

                        # 낙상 경고 표시
                        self._draw_fall_alert(annotated_frame, keypoints, analysis)

        # 낙상 발생 시 처리
        if fall_detected:
            self._handle_fall_event(annotated_frame, timestamp)

        # 통계 표시
        annotated_frame = self._draw_statistics(annotated_frame)

        return annotated_frame, fall_detected, all_analysis

    def _draw_pose(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        keypoint_conf: np.ndarray,
        analysis: dict
    ) -> np.ndarray:
        """포즈 시각화"""
        # 스켈레톤 연결 정의
        skeleton = [
            # 머리
            (0, 1), (0, 2), (1, 3), (2, 4),
            # 몸통
            (5, 6), (5, 11), (6, 12), (11, 12),
            # 팔
            (5, 7), (7, 9), (6, 8), (8, 10),
            # 다리
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        # 낙상 상태에 따라 색상 변경
        is_fall = analysis['is_fall']
        skeleton_color = (0, 0, 255) if is_fall else (0, 255, 0)
        keypoint_color = (0, 0, 255) if is_fall else (255, 0, 0)

        # 스켈레톤 그리기
        for start_idx, end_idx in skeleton:
            if (keypoint_conf[start_idx] > 0.5 and
                keypoint_conf[end_idx] > 0.5):
                start_point = tuple(keypoints[start_idx].astype(int))
                end_point = tuple(keypoints[end_idx].astype(int))
                cv2.line(frame, start_point, end_point, skeleton_color, 2)

        # 키포인트 그리기
        for idx, (kp, conf) in enumerate(zip(keypoints, keypoint_conf)):
            if conf > 0.5:
                x, y = kp.astype(int)
                cv2.circle(frame, (x, y), 4, keypoint_color, -1)

        # 포즈 정보 표시
        if analysis['body_angle'] is not None:
            # 바운딩 박스 계산
            valid_kps = keypoints[keypoint_conf > 0.5]
            if len(valid_kps) > 0:
                x_min = int(valid_kps[:, 0].min())
                y_min = int(valid_kps[:, 1].min())

                # 포즈 타입
                pose_type = analysis['pose_type']
                angle = analysis['body_angle']

                info_text = f"{pose_type.upper()}"
                angle_text = f"Angle: {angle:.1f}deg"

                cv2.putText(
                    frame,
                    info_text,
                    (x_min, y_min - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    skeleton_color,
                    2
                )

                cv2.putText(
                    frame,
                    angle_text,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        return frame

    def _draw_fall_alert(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        analysis: dict
    ):
        """낙상 경고 표시"""
        # 화면 상단에 큰 경고 메시지
        h, w = frame.shape[:2]

        # 배경
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 255), -1)

        # 텍스트
        cv2.putText(
            frame,
            "!!! FALL DETECTED !!!",
            (w // 2 - 200, 50),
            cv2.FONT_HERSHEY_BOLD,
            1.5,
            (255, 255, 255),
            3
        )

        # 낙상 신뢰도
        confidence = analysis['fall_confidence']
        conf_text = f"Confidence: {confidence*100:.0f}%"
        cv2.putText(
            frame,
            conf_text,
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    def _draw_statistics(self, frame: np.ndarray) -> np.ndarray:
        """통계 정보 표시"""
        stats = self.pose_analyzer.get_statistics()

        y_offset = 100
        x_offset = 10

        # 배경
        cv2.rectangle(
            frame,
            (x_offset - 5, y_offset - 25),
            (x_offset + 250, y_offset + 80),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            frame,
            (x_offset - 5, y_offset - 25),
            (x_offset + 250, y_offset + 80),
            (255, 255, 255),
            2
        )

        # 통계
        texts = [
            f"Total Falls: {stats['total_falls']}",
            f"Current Status: {'FALLEN' if stats['currently_fallen'] else 'NORMAL'}",
        ]

        if stats['currently_fallen']:
            duration = stats['fall_duration']
            texts.append(f"Fall Duration: {duration:.1f}s")

        for i, text in enumerate(texts):
            color = (0, 0, 255) if stats['currently_fallen'] else (0, 255, 0)
            cv2.putText(
                frame,
                text,
                (x_offset, y_offset + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return frame

    def _handle_fall_event(self, frame: np.ndarray, timestamp: float):
        """낙상 이벤트 처리"""
        # 녹화 시작
        if self.record_falls and not self.recording:
            self._start_recording(frame)

        # 알림 전송 (쿨다운 고려)
        if self.notifier and (timestamp - self.last_alert_time) > self.alert_cooldown:
            self._send_fall_alert(frame)
            self.last_alert_time = timestamp

    def _start_recording(self, frame: np.ndarray):
        """녹화 시작"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"fall_{timestamp}.mp4"

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(filename),
            fourcc,
            20,
            (w, h)
        )

        self.recording = True
        self.logger.info(f"낙상 녹화 시작: {filename}")

    def _stop_recording(self):
        """녹화 종료"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            self.logger.info("녹화 종료")

    def _send_fall_alert(self, frame: np.ndarray):
        """낙상 알림 전송"""
        stats = self.pose_analyzer.get_statistics()

        message = (
            f"🚨 낙상 감지!\n\n"
            f"시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"총 낙상 횟수: {stats['total_falls']}회\n"
            f"현재 낙상 지속: {stats['fall_duration']:.1f}초"
        )

        # 스크린샷 저장
        screenshot_path = self.output_dir / f"alert_{int(time.time())}.jpg"
        cv2.imwrite(str(screenshot_path), frame)

        # 알림 전송
        self.notifier.send_alert('alert', message, str(screenshot_path))
        self.logger.info("낙상 알림 전송 완료")

    def write_frame(self, frame: np.ndarray):
        """녹화 중이면 프레임 쓰기"""
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def cleanup(self):
        """정리"""
        if self.recording:
            self._stop_recording()


def main():
    parser = argparse.ArgumentParser(description='포즈 추정 기반 낙상 감지 시스템')
    parser.add_argument('--camera', type=str, choices=['usb', 'csi', 'rtsp'], default='usb')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--url', type=str, help='RTSP URL')
    parser.add_argument('--model', type=str, default='yolov8n-pose.pt',
                        help='YOLOv8-pose 모델')

    # 낙상 감지 설정
    parser.add_argument('--fall-angle', type=float, default=45.0,
                        help='낙상 판정 각도 임계값')
    parser.add_argument('--fall-height-ratio', type=float, default=0.3,
                        help='낙상 판정 높이 비율')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='키포인트 신뢰도 임계값')

    # 녹화 설정
    parser.add_argument('--record', action='store_true', default=True,
                        help='낙상 시 자동 녹화')
    parser.add_argument('--output-dir', type=str, default='recordings/falls',
                        help='녹화 파일 저장 디렉토리')

    # 알림 설정
    parser.add_argument('--telegram-token', type=str)
    parser.add_argument('--telegram-chat-id', type=str)
    parser.add_argument('--alert-cooldown', type=int, default=60,
                        help='알림 쿨다운 (초)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 포즈 분석기 초기화
    pose_analyzer = PoseAnalyzer(
        fall_angle_threshold=args.fall_angle,
        fall_height_ratio=args.fall_height_ratio,
        confidence_threshold=args.confidence
    )

    # 알림 설정
    notifier = None
    if args.telegram_token and args.telegram_chat_id:
        notifier = TelegramNotifier(args.telegram_token, args.telegram_chat_id)
        print("✓ 텔레그램 알림 활성화")

    # 낙상 감지 시스템 초기화
    fall_detector = FallDetectionSystem(
        model_path=args.model,
        pose_analyzer=pose_analyzer,
        notifier=notifier,
        record_falls=args.record,
        output_dir=args.output_dir
    )

    if args.alert_cooldown:
        fall_detector.alert_cooldown = args.alert_cooldown

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

    print("\n" + "="*60)
    print("포즈 추정 기반 낙상 감지 시스템 시작")
    print("="*60)
    print(f"모델: {args.model}")
    print(f"낙상 각도 임계값: {args.fall_angle}도")
    print(f"녹화: {'ON' if args.record else 'OFF'}")
    print(f"알림: {'ON' if notifier else 'OFF'}")
    print("\nControls:")
    print("  q: 종료")
    print("  r: 통계 초기화")
    print("  s: 스크린샷 저장")
    print("="*60 + "\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            timestamp = time.time()

            # 낙상 감지 처리
            annotated, fall_detected, analysis = fall_detector.process_frame(
                frame,
                timestamp
            )

            # 녹화
            if args.record:
                fall_detector.write_frame(annotated)

                # 낙상 상태가 해제되면 녹화 종료
                stats = pose_analyzer.get_statistics()
                if not stats['currently_fallen'] and fall_detector.recording:
                    fall_detector._stop_recording()

            # 표시
            cv2.imshow('Fall Detection System', annotated)

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n시스템 종료")
                break
            elif key == ord('r'):
                pose_analyzer.reset_statistics()
                print("\n통계 초기화됨")
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"\n스크린샷 저장: {filename}")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단됨")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        fall_detector.cleanup()
        camera.stop()
        cv2.destroyAllWindows()

        # 최종 통계
        stats = pose_analyzer.get_statistics()
        print("\n" + "="*60)
        print("최종 통계")
        print("="*60)
        print(f"총 낙상 감지: {stats['total_falls']}회")
        print(f"저장 위치: {args.output_dir}")
        print("="*60)


if __name__ == "__main__":
    main()
