# Raspberry Pi YOLO Vision Project

라즈베리파이에서 실시간 객체 탐지를 수행하는 YOLO 기반 비전 시스템

## 주요 기능

- 🎥 **다양한 카메라 지원** (CSI, USB, RTSP)
- 🤖 **YOLOv5/v8/v11 모델** 지원
- 🎯 **실시간 객체 탐지 및 추적**
- 🚨 **침입 감지** 및 실시간 알림
- 📊 **라인 통과 카운팅** (양방향)
- 🅿️ **주차 공간 모니터링**
- 🦺 **안전장비 착용 확인**
- 📱 **텔레그램/이메일 알림**
- ⚡ **라즈베리파이 최적화**

## 시스템 요구사항

### 하드웨어
- Raspberry Pi 4/5 (4GB RAM 이상 권장)
- 카메라 (CSI Camera Module v2/v3 또는 USB 웹캠)
- microSD 카드 (32GB 이상)
- 전원 공급 장치 (5V 3A)

### 소프트웨어
- Raspberry Pi OS (64-bit, Bullseye 이상)
- Python 3.8+
- OpenCV 4.5+

## 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/YawnsDuzin/rasp_vision_yolo.git
cd rasp_vision_yolo
```

### 2. 환경 설정

```bash
# 시스템 패키지 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y python3-pip python3-dev python3-opencv
sudo apt install -y libatlas-base-dev libopenblas-dev
sudo apt install -y libhdf5-dev libhdf5-serial-dev

# CSI 카메라 사용 시
sudo apt install -y python3-picamera2

# Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# Python 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. YOLO 모델 다운로드

```bash
# YOLOv8n 모델 자동 다운로드 (첫 실행 시)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 4. 카메라 테스트

```bash
# USB 카메라
python examples/camera_test.py --camera usb

# CSI 카메라
python examples/camera_test.py --camera csi

# RTSP 스트림
python examples/camera_test.py --camera rtsp --url rtsp://your-camera-ip/stream
```

### 5. 객체 탐지 실행

```bash
# 기본 실행
python src/main.py

# 특정 모델 사용
python src/main.py --model yolov8s.pt

# 신뢰도 임계값 조정
python src/main.py --conf 0.6

# 특정 클래스만 탐지 (사람만)
python src/main.py --classes 0
```

## 프로젝트 구조

```
rasp_vision_yolo/
├── configs/                 # 설정 파일
│   ├── camera_config.yaml
│   ├── model_config.yaml
│   └── detection_zones.yaml
├── models/                  # YOLO 모델 파일
│   └── yolov8n.pt
├── src/                     # 소스 코드
│   ├── camera/             # 카메라 인터페이스
│   ├── detection/          # 객체 탐지 모듈
│   ├── processing/         # 영상 처리
│   ├── utils/              # 유틸리티
│   └── main.py            # 메인 애플리케이션
├── examples/               # 예제 코드
├── tests/                  # 테스트
├── logs/                   # 로그 파일
├── recordings/             # 녹화 파일
├── requirements.txt        # Python 의존성
├── DEVELOPMENT_GUIDE.md   # 상세 개발 가이드
└── README.md              # 프로젝트 README
```

## 설정

### 카메라 설정 (configs/camera_config.yaml)

```yaml
camera:
  type: "usb"  # usb, csi, rtsp
  device_id: 0
  resolution:
    width: 640
    height: 480
  fps: 30

rtsp:
  url: "rtsp://username:password@ip:port/stream"
  buffer_size: 1
```

### 모델 설정 (configs/model_config.yaml)

```yaml
model:
  path: "yolov8n.pt"
  confidence: 0.5
  iou_threshold: 0.45
  classes: null  # null for all classes, or [0, 1, 2] for specific

performance:
  input_size: 640
  half_precision: false
  device: "cpu"
```

## 실용 예제 시나리오

### 🚨 1. 침입 감지 시스템 (보안)

```bash
# 기본 실행
python examples/intrusion_detection.py --camera usb

# 텔레그램 알림 포함
python examples/intrusion_detection.py \
  --camera usb \
  --telegram-token YOUR_TOKEN \
  --telegram-chat-id YOUR_CHAT_ID \
  --save-alerts
```

**기능**:
- ROI 기반 침입 감지
- 실시간 텔레그램 알림
- 침입 이미지 자동 저장
- 통계 추적

**코드 예시**:
```python
from src.detection import IntrusionDetector

# ROI 정의 (다각형)
roi = [(200, 150), (440, 150), (440, 330), (200, 330)]

detector = IntrusionDetector(
    roi_points=roi,
    target_classes=[0],  # 사람만 탐지
    cooldown_seconds=5
)

intrusion, intruders = detector.check_intrusion(detections)
if intrusion and detector.should_alert():
    detector.trigger_alert()
```

---

### 👥 2. 사람 카운팅 (리테일/스마트시티)

```bash
# 기본 실행
python examples/people_counting.py --camera usb

# 정기 리포트 전송
python examples/people_counting.py \
  --camera usb \
  --report-interval 60 \
  --telegram-token YOUR_TOKEN \
  --telegram-chat-id YOUR_CHAT_ID
```

**기능**:
- 양방향 라인 통과 카운팅
- IN/OUT 통계
- 정기 리포트 자동 전송
- 실시간 대시보드

**코드 예시**:
```python
from src.detection import LineCrossingCounter

# 카운팅 라인 정의
counter = LineCrossingCounter(
    line_start=(0, 240),
    line_end=(640, 240),
    target_classes=[0],  # person
    bidirectional=True
)

count_in, count_out, crossed = counter.update(detections)
```

---

### 🅿️ 3. 주차장 모니터링 (스마트시티)

```bash
# 기본 실행
python examples/parking_monitor.py --camera usb

# 설정 파일 사용
python examples/parking_monitor.py \
  --camera rtsp \
  --url rtsp://camera-ip/stream \
  --config configs/parking_config.yaml
```

**기능**:
- 다중 주차 공간 추적
- 실시간 가용 공간 표시
- 점유율 통계
- 예약석/장애인석 관리

**코드 예시**:
```python
from src.detection import ParkingSpaceMonitor

# 주차 공간 정의
parking_spaces = [
    {'id': 1, 'points': [(50, 200), (150, 200), ...], 'reserved': False},
    # ... 추가 공간
]

monitor = ParkingSpaceMonitor(parking_spaces)
space_status = monitor.check_occupancy(vehicles)
stats = monitor.get_statistics()
```

---

### 🦺 4. 안전장비 착용 확인 (제조/건설)

**코드 예시**:
```python
from src.detection import SafetyEquipmentDetector

# 필수 장비 정의
detector = SafetyEquipmentDetector(
    required_equipment=['helmet', 'vest'],
    check_distance_threshold=100
)

# 사람과 장비 분리
persons = [d for d in detections if d['class_name'] == 'person']
equipment = [d for d in detections if d['class_name'] in ['helmet', 'vest']]

# 준수 여부 확인
compliant, violations = detector.check_compliance(persons, equipment)
```

---

### 📱 5. 알림 시스템

```python
from src.utils import TelegramNotifier, MultiNotifier

# 텔레그램 알림
notifier = TelegramNotifier(
    bot_token='YOUR_BOT_TOKEN',
    chat_id='YOUR_CHAT_ID'
)

# 이미지와 함께 알림 전송
notifier.send_alert(
    alert_type='intrusion',
    message='침입자 탐지!\n위치: 정문',
    image_path='alert.jpg'
)

# 다중 채널 통합
multi = MultiNotifier()
multi.add_telegram(token, chat_id)
multi.add_email(smtp_server, smtp_port, sender, password, recipient)
multi.send_alert('alert', '중요 이벤트 발생!')
```

---

### 📊 6. 혼잡도 분석 (스마트시티/이벤트)

```bash
# 기본 실행
python examples/crowd_density.py --camera usb

# 임계값 설정 및 알림
python examples/crowd_density.py \
  --camera usb \
  --threshold-low 5 \
  --threshold-medium 10 \
  --threshold-high 15 \
  --telegram-token YOUR_TOKEN \
  --telegram-chat-id YOUR_CHAT_ID
```

**기능**:
- 다중 영역 혼잡도 모니터링
- 실시간 밀도 분석 (LOW/MEDIUM/HIGH)
- 영역별 평균 인원 추적
- 혼잡도 경고 알림

**활용**: 스마트시티, 이벤트 관리, 쇼핑몰, 지하철역

---

### 🎥 7. 이벤트 기반 자동 녹화

```bash
# 기본 실행 (사람 탐지 시 녹화)
python examples/event_recorder.py --camera usb

# 설정 커스터마이징
python examples/event_recorder.py \
  --camera usb \
  --trigger-classes 0 2 3 \  # person, car, motorcycle
  --min-objects 2 \
  --pre-event 5 \
  --post-event 10
```

**기능**:
- 특정 객체 탐지 시 자동 녹화
- 이벤트 전후 프레임 버퍼링
- 최대 파일 크기 관리
- 녹화 파일 자동 저장

**활용**: 보안 CCTV, 증거 수집, 이벤트 기록

---

### 🎯 8. 고급 객체 추적

```bash
# 기본 실행
python examples/object_tracker.py --camera usb

# 히트맵 포함
python examples/object_tracker.py \
  --camera usb \
  --dwell-threshold 3 \
  --show-heatmap
```

**기능**:
- 객체별 고유 ID 추적
- 이동 경로 시각화
- 체류 시간 측정
- 이동 속도 계산
- 히트맵 생성 (방문 밀도)

**활용**: 리테일 동선 분석, 행동 패턴 연구, 보안 모니터링

---

### 🏥 9. 낙상 감지 시스템 (의료/헬스케어)

```bash
# 기본 실행
python examples/fall_detection.py --camera usb

# 설정 커스터마이징 및 알림
python examples/fall_detection.py \
  --camera usb \
  --fall-angle 45 \
  --fall-height-ratio 0.3 \
  --record \
  --telegram-token YOUR_TOKEN \
  --telegram-chat-id YOUR_CHAT_ID
```

**기능**:
- YOLOv8-pose 기반 실시간 포즈 추정
- 17개 관절점(keypoints) 추적
- 다중 낙상 감지 알고리즘:
  - 몸통 각도 분석 (수평 자세 감지)
  - 머리 높이 비율 분석
  - 급격한 자세 변화 감지
- 자동 녹화 및 알림
- 포즈 타입 분류 (서있음/앉음/누움/낙상)

**코드 예시**:
```python
from src.detection import PoseAnalyzer

# 포즈 분석기 초기화
analyzer = PoseAnalyzer(
    fall_angle_threshold=45.0,  # 낙상 판정 각도
    fall_height_ratio=0.3,      # 낙상 판정 높이 비율
    confidence_threshold=0.5    # 키포인트 신뢰도
)

# 포즈 분석
analysis = analyzer.analyze_pose(keypoints, keypoint_conf)

if analysis['is_fall']:
    print(f"낙상 감지! 신뢰도: {analysis['fall_confidence']*100:.0f}%")
    print(f"포즈 타입: {analysis['pose_type']}")
    print(f"몸통 각도: {analysis['body_angle']:.1f}도")
```

**활용**: 요양원, 병원, 독거노인 모니터링, 재활 센터

---

## 성능 최적화

### 라즈베리파이 4 예상 성능

| 모델 | 입력 크기 | FPS | mAP |
|------|----------|-----|-----|
| YOLOv5n | 320x320 | 8-10 | 28.0 |
| YOLOv8n | 640x640 | 4-6 | 37.3 |
| YOLOv8s | 640x640 | 2-3 | 44.9 |

### 최적화 팁

1. **경량 모델 사용**: YOLOv8n 권장
2. **입력 크기 감소**: 640 → 320
3. **프레임 스킵**: 매 3프레임마다 처리
4. **ROI 처리**: 관심 영역만 처리
5. **ONNX 변환**: 추론 속도 20-30% 향상

```bash
# ONNX 변환
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
```

## 시스템 서비스 등록

```bash
# systemd 서비스 파일 생성
sudo nano /etc/systemd/system/yolo-vision.service

# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable yolo-vision.service
sudo systemctl start yolo-vision.service

# 상태 확인
sudo systemctl status yolo-vision.service
```

## 트러블슈팅

### 카메라가 인식되지 않을 때

```bash
# USB 카메라 확인
ls /dev/video*
v4l2-ctl --list-devices

# CSI 카메라 활성화
sudo raspi-config
# Interface Options > Camera > Enable
```

### 메모리 부족 오류

```bash
# GPU 메모리 증가
sudo raspi-config
# Performance Options > GPU Memory > 256

# 스왑 크기 증가
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 낮은 FPS

1. 경량 모델 사용 (YOLOv8n 또는 YOLOv5n)
2. 입력 해상도 감소 (320x320)
3. 신뢰도 임계값 증가 (0.6 이상)
4. 프레임 스킵 활성화

## 활용 사례

- 🏠 홈 보안 시스템
- 🚗 주차장 관리
- 🌾 스마트팜 모니터링
- 🏭 제조 품질 검사
- 🐾 야생동물 관찰
- 🚦 교통 분석

## 참고 자료

- [상세 개발 가이드](DEVELOPMENT_GUIDE.md)
- [Ultralytics 공식 문서](https://docs.ultralytics.com)
- [Raspberry Pi 카메라 문서](https://www.raspberrypi.com/documentation/accessories/camera.html)

## 라이선스

MIT License

## 기여

이슈 및 풀 리퀘스트는 환영합니다!

## 문의

프로젝트 관련 문의사항은 이슈로 남겨주세요.

---

**개발**: YawnsDuzin
**업데이트**: 2025-11-12
