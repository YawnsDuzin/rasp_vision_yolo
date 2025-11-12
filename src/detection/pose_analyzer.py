"""
Pose Analyzer
포즈 분석 모듈 - 낙상 감지, 자세 분석
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from collections import deque
import time


class PoseAnalyzer:
    """포즈 분석 및 낙상 감지"""

    # COCO 키포인트 인덱스
    KEYPOINTS = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }

    def __init__(
        self,
        fall_angle_threshold: float = 45.0,
        fall_height_ratio: float = 0.3,
        confidence_threshold: float = 0.5,
        history_size: int = 30
    ):
        """
        Initialize pose analyzer

        Args:
            fall_angle_threshold: 낙상 판정 각도 임계값 (도)
            fall_height_ratio: 몸통 높이 대비 임계값
            confidence_threshold: 키포인트 신뢰도 임계값
            history_size: 포즈 히스토리 크기
        """
        self.fall_angle_threshold = fall_angle_threshold
        self.fall_height_ratio = fall_height_ratio
        self.confidence_threshold = confidence_threshold

        # 포즈 히스토리
        self.pose_history = deque(maxlen=history_size)

        # 낙상 감지 상태
        self.fall_detected = False
        self.fall_start_time = None
        self.total_falls = 0

        self.logger = logging.getLogger(__name__)

    def analyze_pose(self, keypoints: np.ndarray, keypoint_conf: np.ndarray) -> Dict:
        """
        포즈 분석

        Args:
            keypoints: 키포인트 좌표 (17, 2) - (x, y)
            keypoint_conf: 키포인트 신뢰도 (17,)

        Returns:
            분석 결과 딕셔너리
        """
        result = {
            'is_fall': False,
            'fall_confidence': 0.0,
            'body_angle': None,
            'body_height_ratio': None,
            'pose_type': 'unknown',
            'analysis': {}
        }

        # 주요 키포인트 추출
        head = self._get_keypoint(keypoints, keypoint_conf, 'nose')
        left_shoulder = self._get_keypoint(keypoints, keypoint_conf, 'left_shoulder')
        right_shoulder = self._get_keypoint(keypoints, keypoint_conf, 'right_shoulder')
        left_hip = self._get_keypoint(keypoints, keypoint_conf, 'left_hip')
        right_hip = self._get_keypoint(keypoints, keypoint_conf, 'right_hip')
        left_knee = self._get_keypoint(keypoints, keypoint_conf, 'left_knee')
        right_knee = self._get_keypoint(keypoints, keypoint_conf, 'right_knee')

        # 키포인트가 충분히 탐지되지 않으면 분석 불가
        valid_points = sum([
            head is not None,
            left_shoulder is not None or right_shoulder is not None,
            left_hip is not None or right_hip is not None
        ])

        if valid_points < 2:
            result['pose_type'] = 'insufficient_keypoints'
            return result

        # 어깨 중심점
        shoulder_center = self._get_center_point(left_shoulder, right_shoulder)

        # 엉덩이 중심점
        hip_center = self._get_center_point(left_hip, right_hip)

        # 무릎 중심점
        knee_center = self._get_center_point(left_knee, right_knee)

        # 몸통 각도 계산
        if shoulder_center is not None and hip_center is not None:
            body_angle = self._calculate_angle(shoulder_center, hip_center)
            result['body_angle'] = body_angle
        else:
            body_angle = None

        # 몸통 높이 비율 계산
        if head is not None and hip_center is not None:
            body_height_ratio = self._calculate_height_ratio(
                head, shoulder_center, hip_center
            )
            result['body_height_ratio'] = body_height_ratio
        else:
            body_height_ratio = None

        # 낙상 감지 로직
        fall_indicators = []

        # 1. 몸통이 수평에 가까운지 (각도가 임계값보다 작음)
        if body_angle is not None:
            if abs(body_angle - 90) > (90 - self.fall_angle_threshold):
                fall_indicators.append('horizontal_body')

        # 2. 머리 위치가 낮은지
        if body_height_ratio is not None:
            if body_height_ratio < self.fall_height_ratio:
                fall_indicators.append('low_head_position')

        # 3. 급격한 자세 변화
        if len(self.pose_history) > 5:
            if self._detect_sudden_change(keypoints):
                fall_indicators.append('sudden_movement')

        # 낙상 판정
        fall_confidence = len(fall_indicators) / 3.0
        is_fall = fall_confidence >= 0.6  # 3개 중 2개 이상

        result['is_fall'] = is_fall
        result['fall_confidence'] = fall_confidence
        result['analysis'] = {
            'fall_indicators': fall_indicators,
            'shoulder_center': shoulder_center,
            'hip_center': hip_center,
            'knee_center': knee_center
        }

        # 포즈 타입 분류
        result['pose_type'] = self._classify_pose_type(
            body_angle, body_height_ratio, is_fall
        )

        # 히스토리 저장
        self.pose_history.append({
            'timestamp': time.time(),
            'keypoints': keypoints.copy(),
            'body_angle': body_angle,
            'is_fall': is_fall
        })

        # 낙상 상태 업데이트
        if is_fall and not self.fall_detected:
            self.fall_detected = True
            self.fall_start_time = time.time()
            self.total_falls += 1
            self.logger.warning(f"낙상 감지! (총 {self.total_falls}회)")
        elif not is_fall and self.fall_detected:
            fall_duration = time.time() - self.fall_start_time
            self.logger.info(f"낙상 상태 해제 (지속 시간: {fall_duration:.1f}초)")
            self.fall_detected = False

        return result

    def _get_keypoint(
        self,
        keypoints: np.ndarray,
        keypoint_conf: np.ndarray,
        name: str
    ) -> Optional[Tuple[float, float]]:
        """키포인트 추출 (신뢰도 체크)"""
        idx = self.KEYPOINTS[name]
        if keypoint_conf[idx] >= self.confidence_threshold:
            return tuple(keypoints[idx])
        return None

    def _get_center_point(
        self,
        point1: Optional[Tuple],
        point2: Optional[Tuple]
    ) -> Optional[Tuple[float, float]]:
        """두 점의 중심점 계산"""
        if point1 is not None and point2 is not None:
            return (
                (point1[0] + point2[0]) / 2,
                (point1[1] + point2[1]) / 2
            )
        elif point1 is not None:
            return point1
        elif point2 is not None:
            return point2
        return None

    def _calculate_angle(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """두 점을 연결하는 선의 수평선 대비 각도 (도)"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        # 수직 = 90도, 수평 = 0도
        return abs(angle_deg)

    def _calculate_height_ratio(
        self,
        head: Tuple[float, float],
        shoulder: Optional[Tuple[float, float]],
        hip: Tuple[float, float]
    ) -> float:
        """머리-어깨-엉덩이 높이 비율"""
        if shoulder is not None:
            total_height = abs(hip[1] - head[1])
            upper_height = abs(shoulder[1] - head[1])
            if total_height > 0:
                return upper_height / total_height
        return 1.0

    def _detect_sudden_change(self, current_keypoints: np.ndarray) -> bool:
        """급격한 자세 변화 감지"""
        if len(self.pose_history) < 5:
            return False

        # 최근 5개 프레임의 평균 위치
        recent_poses = list(self.pose_history)[-5:]
        avg_keypoints = np.mean([p['keypoints'] for p in recent_poses], axis=0)

        # 현재 키포인트와 평균의 차이
        diff = np.linalg.norm(current_keypoints - avg_keypoints, axis=1)
        avg_diff = np.mean(diff)

        # 임계값 (픽셀)
        threshold = 50

        return avg_diff > threshold

    def _classify_pose_type(
        self,
        body_angle: Optional[float],
        body_height_ratio: Optional[float],
        is_fall: bool
    ) -> str:
        """포즈 타입 분류"""
        if is_fall:
            return 'fallen'

        if body_angle is None:
            return 'unknown'

        # 서있는 자세
        if 60 <= body_angle <= 90:
            return 'standing'

        # 앉은 자세
        if 30 <= body_angle < 60:
            return 'sitting'

        # 누운 자세
        if body_angle < 30:
            return 'lying'

        return 'other'

    def get_statistics(self) -> Dict:
        """통계 정보"""
        return {
            'total_falls': self.total_falls,
            'currently_fallen': self.fall_detected,
            'fall_duration': (
                time.time() - self.fall_start_time
                if self.fall_detected else 0
            )
        }

    def reset_statistics(self):
        """통계 초기화"""
        self.total_falls = 0
        self.fall_detected = False
        self.fall_start_time = None
        self.pose_history.clear()
