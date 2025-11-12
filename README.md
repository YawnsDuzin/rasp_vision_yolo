# Raspberry Pi YOLO Vision Project

라즈베리파이에서 실시간 객체 탐지를 수행하는 YOLO 기반 비전 시스템

## 주요 기능

- 🎥 다양한 카메라 지원 (CSI, USB, RTSP)
- 🤖 YOLOv5/v8/v11 모델 지원
- 🎯 실시간 객체 탐지 및 추적
- 🚨 침입 감지 및 알림
- 📊 객체 카운팅 및 통계
- ⚡ 라즈베리파이 최적화

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

## 주요 기능 예시

### 1. 침입 감지

```python
from src.detection.intrusion_detector import IntrusionDetector

# ROI 정의 (다각형)
roi = [(100, 100), (500, 100), (500, 400), (100, 400)]

detector = IntrusionDetector(model_path='yolov8n.pt', roi=roi)
# ... 카메라 루프에서 사용
```

### 2. 객체 카운팅

```python
from src.detection.line_counter import LineCounter

# 카운팅 라인 정의
line = (0, 300, 640, 300)  # (x1, y1, x2, y2)

counter = LineCounter(line_position=line)
# ... 추적과 함께 사용
```

### 3. 알림 설정

```python
from src.utils.notifier import TelegramNotifier

notifier = TelegramNotifier(
    bot_token='YOUR_BOT_TOKEN',
    chat_id='YOUR_CHAT_ID'
)

# 탐지 시 알림 전송
notifier.send_alert("침입자 탐지!", image_path='alert.jpg')
```

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
