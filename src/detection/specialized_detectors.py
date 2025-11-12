"""
Safety Equipment Detector
안전장비 착용 확인 - 제조 및 건설 분야 활용
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple
from collections import defaultdict


class SafetyEquipmentDetector:
    """
    안전장비 착용 확인 시스템
    헬멧, 안전조끼, 안전화 등의 착용 여부 확인
    """

    def __init__(
        self,
        required_equipment: List[str] = None,
        check_distance_threshold: int = 100
    ):
        """
        Initialize safety equipment detector

        Args:
            required_equipment: 필수 안전장비 리스트 ['helmet', 'vest', 'boots']
            check_distance_threshold: 사람과 장비 간 최대 거리 (픽셀)
        """
        self.required_equipment = required_equipment or ['helmet']
        self.check_distance = check_distance_threshold

        # 장비별 클래스 매핑 (커스텀 모델 학습 필요)
        self.equipment_classes = {
            'helmet': ['helmet', 'hardhat', 'hard-hat'],
            'vest': ['vest', 'safety-vest', 'high-vis'],
            'boots': ['boots', 'safety-boots']
        }

        self.violations = []
        self.violation_count = 0

        self.logger = logging.getLogger(__name__)

    def check_compliance(
        self,
        persons: List[dict],
        equipment: List[dict]
    ) -> Tuple[List[dict], List[dict]]:
        """
        안전장비 착용 여부 확인

        Args:
            persons: 탐지된 사람 리스트
            equipment: 탐지된 장비 리스트

        Returns:
            (준수자 리스트, 위반자 리스트)
        """
        compliant = []
        violations = []

        for person in persons:
            person_center = person['center']
            person_bbox = person['bbox']

            # 각 필수 장비 확인
            equipment_status = {}

            for req_equipment in self.required_equipment:
                has_equipment = self._check_person_has_equipment(
                    person_bbox,
                    person_center,
                    equipment,
                    req_equipment
                )
                equipment_status[req_equipment] = has_equipment

            # 모든 장비 착용 여부
            if all(equipment_status.values()):
                compliant.append({
                    **person,
                    'equipment_status': equipment_status,
                    'compliant': True
                })
            else:
                violations.append({
                    **person,
                    'equipment_status': equipment_status,
                    'compliant': False,
                    'missing': [k for k, v in equipment_status.items() if not v]
                })
                self.violation_count += 1

        return compliant, violations

    def _check_person_has_equipment(
        self,
        person_bbox: Tuple[int, int, int, int],
        person_center: Tuple[int, int],
        equipment_list: List[dict],
        equipment_type: str
    ) -> bool:
        """
        특정 사람이 특정 장비를 착용했는지 확인

        Args:
            person_bbox: 사람 바운딩 박스
            person_center: 사람 중심점
            equipment_list: 장비 탐지 리스트
            equipment_type: 장비 유형

        Returns:
            착용 여부
        """
        px1, py1, px2, py2 = person_bbox

        for equip in equipment_list:
            equip_name = equip['class_name'].lower()

            # 장비 클래스 매칭
            if not any(
                eq_class in equip_name
                for eq_class in self.equipment_classes.get(equipment_type, [])
            ):
                continue

            equip_center = equip['center']

            # 거리 계산
            distance = np.sqrt(
                (person_center[0] - equip_center[0]) ** 2 +
                (person_center[1] - equip_center[1]) ** 2
            )

            # 사람 박스 내부 또는 가까운 거리에 있는지 확인
            ex1, ey1, ex2, ey2 = equip['bbox']

            # 헬멧은 사람 상단에 있어야 함
            if equipment_type == 'helmet':
                if (
                    ex1 >= px1 and ex2 <= px2 and
                    ey1 >= py1 and ey2 <= py1 + (py2 - py1) * 0.3
                ):
                    return True

            # 조끼는 사람 중앙부에 있어야 함
            elif equipment_type == 'vest':
                if (
                    ex1 >= px1 and ex2 <= px2 and
                    ey1 >= py1 + (py2 - py1) * 0.2 and
                    ey2 <= py1 + (py2 - py1) * 0.7
                ):
                    return True

            # 일반적인 거리 기반 체크
            elif distance < self.check_distance:
                return True

        return False

    def draw_compliance(
        self,
        frame: np.ndarray,
        compliant: List[dict],
        violations: List[dict]
    ) -> np.ndarray:
        """
        준수/위반 상태 시각화

        Args:
            frame: 입력 프레임
            compliant: 준수자 리스트
            violations: 위반자 리스트
        """
        # 준수자 (녹색)
        for person in compliant:
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "✓ SAFE",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # 위반자 (빨간색)
        for person in violations:
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # 누락된 장비 표시
            missing = ', '.join(person['missing'])
            label = f"⚠ NO {missing.upper()}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        return frame

    def draw_statistics(
        self,
        frame: np.ndarray,
        compliant_count: int,
        violation_count: int,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """통계 정보 표시"""
        x, y = position

        total = compliant_count + violation_count
        compliance_rate = (
            (compliant_count / total * 100) if total > 0 else 0
        )

        # 배경
        cv2.rectangle(frame, (x - 5, y - 25), (x + 300, y + 75), (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 5, y - 25), (x + 300, y + 75), (255, 255, 255), 2)

        # 통계
        texts = [
            f"Compliant: {compliant_count}",
            f"Violations: {violation_count}",
            f"Compliance Rate: {compliance_rate:.1f}%"
        ]

        for i, text in enumerate(texts):
            color = (0, 255, 0) if i == 0 else (0, 0, 255) if i == 1 else (255, 255, 255)
            cv2.putText(
                frame,
                text,
                (x, y + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return frame


class ParkingSpaceMonitor:
    """
    주차 공간 모니터링 - 스마트시티 분야 활용
    주차 구역별 차량 점유 상태 추적
    """

    def __init__(self, parking_spaces: List[dict]):
        """
        Initialize parking monitor

        Args:
            parking_spaces: 주차 공간 리스트
                [{'id': 1, 'points': [(x1,y1), (x2,y2), ...], 'reserved': False}, ...]
        """
        self.parking_spaces = parking_spaces
        self.occupancy_history = defaultdict(list)
        self.logger = logging.getLogger(__name__)

    def check_occupancy(self, vehicles: List[dict]) -> Dict[int, dict]:
        """
        주차 공간 점유 상태 확인

        Args:
            vehicles: 탐지된 차량 리스트 (class: car, truck, bus 등)

        Returns:
            공간별 점유 정보 딕셔너리
        """
        space_status = {}

        for space in self.parking_spaces:
            space_id = space['id']
            space_polygon = np.array(space['points'], dtype=np.int32)

            # 이 공간에 차량이 있는지 확인
            occupied = False
            vehicle_in_space = None

            for vehicle in vehicles:
                center = vehicle['center']

                if cv2.pointPolygonTest(space_polygon, center, False) >= 0:
                    occupied = True
                    vehicle_in_space = vehicle
                    break

            space_status[space_id] = {
                'occupied': occupied,
                'reserved': space.get('reserved', False),
                'vehicle': vehicle_in_space,
                'points': space['points']
            }

            # 히스토리 기록
            self.occupancy_history[space_id].append(occupied)
            if len(self.occupancy_history[space_id]) > 100:
                self.occupancy_history[space_id].pop(0)

        return space_status

    def draw_spaces(
        self,
        frame: np.ndarray,
        space_status: Dict[int, dict]
    ) -> np.ndarray:
        """주차 공간 시각화"""
        for space_id, status in space_status.items():
            points = np.array(status['points'], dtype=np.int32)

            # 색상 결정
            if status['occupied']:
                color = (0, 0, 255)  # 빨강 (점유)
                status_text = "OCCUPIED"
            elif status['reserved']:
                color = (0, 255, 255)  # 노랑 (예약)
                status_text = "RESERVED"
            else:
                color = (0, 255, 0)  # 녹색 (비어있음)
                status_text = "AVAILABLE"

            # 공간 그리기
            cv2.polylines(frame, [points], True, color, 2)

            # 반투명 오버레이
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # 텍스트
            center_x = int(np.mean([p[0] for p in points]))
            center_y = int(np.mean([p[1] for p in points]))

            cv2.putText(
                frame,
                f"#{space_id}",
                (center_x - 20, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                status_text,
                (center_x - 40, center_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return frame

    def get_statistics(self) -> dict:
        """통계 정보"""
        total_spaces = len(self.parking_spaces)
        occupied_spaces = sum(
            1 for history in self.occupancy_history.values()
            if history and history[-1]
        )
        available_spaces = total_spaces - occupied_spaces
        occupancy_rate = (occupied_spaces / total_spaces * 100) if total_spaces > 0 else 0

        return {
            'total': total_spaces,
            'occupied': occupied_spaces,
            'available': available_spaces,
            'occupancy_rate': occupancy_rate
        }
