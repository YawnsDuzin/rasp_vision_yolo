# 라즈베리파이 YOLO 비전 프로젝트 개발 가이드

## 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [YOLO 모델 비교](#yolo-모델-비교)
3. [카메라 스트리밍 방식](#카메라-스트리밍-방식)
4. [개발 환경 설정](#개발-환경-설정)
5. [개발 진행 절차](#개발-진행-절차)
6. [YOLO 활용 기능](#yolo-활용-기능)
7. [산업별 활용 사례](#산업별-활용-사례)
8. [성능 최적화](#성능-최적화)

---

## 프로젝트 개요

라즈베리파이에서 실시간 객체 탐지를 위한 YOLO 기반 비전 시스템 구축

### 기술 스택
- **언어**: Python 3.7+
- **딥러닝 프레임워크**: PyTorch, ONNX Runtime
- **객체 탐지**: YOLOv5, YOLOv8, YOLOv11
- **영상 처리**: OpenCV
- **하드웨어**: Raspberry Pi 4/5 권장 (4GB+ RAM)

---

## YOLO 모델 비교

### YOLOv5 (Ultralytics)
**출시**: 2020년

**특징**:
- PyTorch 기반으로 가장 안정적
- 모델 크기: YOLOv5n (1.9MB) ~ YOLOv5x (166MB)
- 라즈베리파이에서 검증된 성능
- 풍부한 커뮤니티 지원

**라즈베리파이 성능**:
- YOLOv5n: ~5-10 FPS (640x640)
- YOLOv5s: ~3-5 FPS (640x640)

**장점**:
- 경량화 모델 (YOLOv5n) 우수
- 문서화 및 예제 풍부
- export 지원 (ONNX, TFLite, TensorRT)

**단점**:
- 최신 기술 대비 정확도 약간 낮음

### YOLOv8 (Ultralytics)
**출시**: 2023년 1월

**특징**:
- YOLOv5 개선 버전
- 모델 크기: YOLOv8n (6.2MB) ~ YOLOv8x (136MB)
- 개선된 아키텍처 (C2f 모듈)
- Task 통합 (Detection, Segmentation, Classification, Pose)

**라즈베리파이 성능**:
- YOLOv8n: ~4-8 FPS (640x640)
- YOLOv8s: ~2-4 FPS (640x640)

**장점**:
- YOLOv5 대비 5-10% 정확도 향상
- 다양한 태스크 지원
- 사용하기 쉬운 CLI/Python API
- 활발한 업데이트

**단점**:
- YOLOv5 대비 약간 무거움

### YOLOv11 (Ultralytics)
**출시**: 2024년 9월

**특징**:
- 최신 YOLO 버전
- 모델 크기: YOLOv11n (5.2MB) ~ YOLOv11x (143MB)
- C3k2 모듈, SPPF 개선
- 향상된 정확도와 속도

**라즈베리파이 성능**:
- YOLOv11n: ~4-7 FPS (640x640) - 실험적

**장점**:
- 최고 정확도
- 파라미터 효율성 개선
- 최신 기술 적용

**단점**:
- 라즈베리파이 최적화 검증 부족
- 일부 라이브러리 호환성 이슈 가능

### YOLO-NAS (Deci AI)
**출시**: 2023년

**특징**:
- Neural Architecture Search로 설계
- 모델 크기: YOLO-NAS-S (~50MB)
- 상업적 사용 제한 있음

**장점**:
- 동일 크기 대비 높은 정확도

**단점**:
- 라이센스 이슈 (상업용 제한)
- 라즈베리파이 최적화 부족

### 라즈베리파이 권장 모델

| 우선순위 | 모델 | 이유 |
|---------|------|------|
| 1순위 | **YOLOv8n** | 성능/정확도 밸런스 최고, 최신 기능 |
| 2순위 | **YOLOv5n** | 가장 경량, 검증된 안정성 |
| 3순위 | **YOLOv8s** | 정확도 중시 시 |

---

## 카메라 스트리밍 방식

### 1. 라즈베리파이 전용 카메라 (CSI)

**지원 모델**:
- Camera Module v2 (8MP)
- Camera Module v3 (12MP)
- HQ Camera (12.3MP)

**장점**:
- 낮은 지연시간 (latency)
- CPU 부하 낮음 (GPU 가속 사용)
- 고해상도 지원
- 저렴한 가격

**단점**:
- CSI 포트 필요
- 케이블 길이 제한 (최대 1m)

**Python 구현**:

```python
# picamera2 사용 (권장 - Raspberry Pi OS Bullseye+)
from picamera2 import Picamera2
import cv2

def init_picamera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    return picam2

def get_frame(picam2):
    frame = picam2.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# 또는 picamera (구 버전)
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

def init_legacy_picamera():
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(640, 480))
    time.sleep(0.1)
    return camera, rawCapture
```

### 2. USB 카메라

**지원 장치**:
- 일반 USB 웹캠
- Logitech C270, C920 등

**장점**:
- 쉬운 연결 (플러그 앤 플레이)
- 긴 케이블 가능
- 다양한 선택지

**단점**:
- CPU 부하 높음
- USB 대역폭 제한
- 라즈베리파이 전원 부족 가능

**Python 구현**:

```python
import cv2

def init_usb_camera(device_id=0):
    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # V4L2 백엔드 사용 (리눅스 최적화)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

# 다중 카메라
cap1 = cv2.VideoCapture(0)  # /dev/video0
cap2 = cv2.VideoCapture(1)  # /dev/video1
```

### 3. RTSP 네트워크 카메라

**지원 장치**:
- IP 카메라
- NVR/DVR 시스템
- 스마트폰 (IP Webcam 앱)

**장점**:
- 원격 위치 카메라 가능
- 고품질 카메라 활용
- 무선 연결

**단점**:
- 네트워크 지연시간
- 대역폭 요구
- 설정 복잡

**Python 구현**:

```python
import cv2

def init_rtsp_camera(rtsp_url):
    """
    rtsp_url 예시:
    - rtsp://username:password@192.168.1.100:554/stream1
    - rtsp://admin:admin@camera.local/live
    """
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # 버퍼 최소화 (지연 감소)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap

def get_frame_with_timeout(cap, timeout=5):
    import time
    start = time.time()
    while (time.time() - start) < timeout:
        ret, frame = cap.read()
        if ret:
            return frame
    return None

# RTSP 재연결 로직
def create_robust_rtsp_stream(rtsp_url, max_retries=5):
    retries = 0
    while retries < max_retries:
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            return cap
        retries += 1
        time.sleep(2 ** retries)
    raise ConnectionError("RTSP 연결 실패")
```

### 카메라 선택 가이드

| 용도 | 권장 카메라 | 이유 |
|------|------------|------|
| 실시간 객체 탐지 | CSI 카메라 | 낮은 지연, 높은 FPS |
| 프로토타입/테스트 | USB 카메라 | 쉬운 설치 |
| 원격 모니터링 | RTSP 카메라 | 유연한 배치 |
| 고품질 영상 | CSI HQ 카메라 | 최고 화질 |

---

## 개발 환경 설정

### 1. 라즈베리파이 OS 설치

```bash
# Raspberry Pi OS Lite 64-bit 권장
# Raspberry Pi Imager로 설치
```

### 2. 시스템 업데이트

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-dev
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libatlas-base-dev libopenblas-dev
```

### 3. Python 가상환경 설정

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 기본 패키지 설치
pip install --upgrade pip
pip install numpy opencv-python
```

### 4. YOLO 설치

#### YOLOv8 설치 (권장)

```bash
pip install ultralytics

# 의존성
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# 또는 경량화 버전
pip install onnxruntime  # ONNX로 변환 후 사용
```

#### YOLOv5 설치

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 5. 카메라 설정

#### CSI 카메라

```bash
# picamera2 설치 (Raspberry Pi OS Bullseye+)
sudo apt install -y python3-picamera2

# 또는 legacy picamera
pip install "picamera[array]"

# 카메라 활성화
sudo raspi-config
# Interface Options > Camera > Enable
```

#### USB 카메라

```bash
# 카메라 확인
ls /dev/video*

# v4l-utils 설치
sudo apt install v4l-utils

# 카메라 정보 확인
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

#### RTSP 카메라

```bash
# GStreamer 설치 (RTSP 성능 향상)
sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-*

# FFmpeg 설치
sudo apt install -y ffmpeg
```

---

## 개발 진행 절차

### Phase 1: 기본 카메라 스트리밍 (1-2일)

**목표**: 카메라로부터 영상 수신 및 표시

**작업**:
1. 카메라 드라이버 설치 및 테스트
2. OpenCV로 프레임 캡처
3. 프레임 전처리 (리사이즈, 포맷 변환)
4. FPS 측정 구현

**코드 예시**:
```python
# camera_test.py
import cv2
import time

cap = cv2.VideoCapture(0)
fps_counter = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fps_counter += 1
    elapsed = time.time() - start_time

    if elapsed > 1.0:
        fps = fps_counter / elapsed
        print(f"FPS: {fps:.2f}")
        fps_counter = 0
        start_time = time.time()

    cv2.imshow('Camera Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Phase 2: YOLO 모델 통합 (2-3일)

**목표**: YOLO 모델 로드 및 추론

**작업**:
1. YOLO 모델 다운로드
2. 모델 추론 파이프라인 구축
3. 바운딩 박스 그리기
4. 클래스 레이블 표시

**코드 예시**:
```python
# yolo_detector.py
from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame, conf_threshold=0.5):
        results = self.model(frame, conf=conf_threshold, verbose=False)
        return results[0]

    def draw_boxes(self, frame, results):
        boxes = results.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{results.names[cls]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

# 사용
detector = YOLODetector('yolov8n.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.detect(frame)
    frame = detector.draw_boxes(frame, results)

    cv2.imshow('YOLO Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Phase 3: 성능 최적화 (3-5일)

**목표**: 라즈베리파이에서 실시간 처리

**작업**:
1. 입력 해상도 최적화 (320x320 또는 640x640)
2. ONNX 변환으로 추론 속도 향상
3. 멀티스레딩 (카메라 읽기 / 추론 분리)
4. Frame skipping 구현

**ONNX 변환**:
```python
# ONNX 변환
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='onnx', simplify=True)

# ONNX Runtime 사용
import onnxruntime as ort
import numpy as np

class ONNXDetector:
    def __init__(self, model_path='yolov8n.onnx'):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def preprocess(self, frame):
        # 전처리 로직
        img = cv2.resize(frame, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32) / 255.0
        return img

    def detect(self, frame):
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs
```

**멀티스레딩**:
```python
# threaded_camera.py
from threading import Thread
import cv2

class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()
```

### Phase 4: 기능 구현 (5-10일)

**목표**: 실제 사용 가능한 애플리케이션 개발

**작업**:
1. 특정 객체 추적
2. 침입 감지 (ROI 설정)
3. 객체 카운팅
4. 알림 시스템 (이메일, 텔레그램)
5. 로깅 및 데이터 저장

**예시 - 침입 감지**:
```python
# intrusion_detector.py
import cv2
from ultralytics import YOLO

class IntrusionDetector:
    def __init__(self, model_path='yolov8n.pt', roi=None):
        self.model = YOLO(model_path)
        self.roi = roi  # [(x1,y1), (x2,y2), ...]
        self.alert_triggered = False

    def point_in_roi(self, point):
        x, y = point
        return cv2.pointPolygonTest(np.array(self.roi), (x, y), False) >= 0

    def detect_intrusion(self, frame):
        results = self.model(frame, classes=[0], conf=0.5)  # person only

        intrusions = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1+x2)//2, (y1+y2)//2)

            if self.point_in_roi(center):
                intrusions.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'conf': float(box.conf[0])
                })

        if intrusions and not self.alert_triggered:
            self.trigger_alert(intrusions)
            self.alert_triggered = True
        elif not intrusions:
            self.alert_triggered = False

        return intrusions

    def trigger_alert(self, intrusions):
        # 알림 전송 로직
        print(f"침입 감지! {len(intrusions)}명 탐지됨")
        # send_telegram_alert(intrusions)
        # send_email_alert(intrusions)
```

### Phase 5: 배포 및 모니터링 (2-3일)

**목표**: 안정적인 운영 환경 구축

**작업**:
1. systemd 서비스 등록
2. 로그 관리
3. 자동 재시작 설정
4. 원격 모니터링 대시보드

**systemd 서비스**:
```bash
# /etc/systemd/system/yolo-vision.service
[Unit]
Description=YOLO Vision Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/rasp_vision_yolo
ExecStart=/home/pi/rasp_vision_yolo/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## YOLO 활용 기능

### 1. 객체 탐지 (Object Detection)

**기능**: 이미지에서 객체의 위치와 클래스 식별

**활용**:
- 사람, 차량, 동물 탐지
- 물체 카운팅
- 침입 감지

**코드**:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model('image.jpg')

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"탐지: {result.names[cls]}, 신뢰도: {conf:.2f}")
```

### 2. 객체 추적 (Object Tracking)

**기능**: 프레임 간 객체 추적

**활용**:
- 사람 동선 추적
- 차량 속도 측정
- 행동 분석

**코드**:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# ByteTrack, BoT-SORT 지원
results = model.track('video.mp4', tracker='bytetrack.yaml', persist=True)

for result in results:
    boxes = result.boxes
    for box in boxes:
        track_id = int(box.id[0]) if box.id is not None else None
        print(f"객체 ID: {track_id}")
```

### 3. 인스턴스 분할 (Instance Segmentation)

**기능**: 픽셀 단위 객체 구분

**활용**:
- 정확한 객체 영역 추출
- 배경 제거
- 객체 크기 측정

**코드**:
```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # segmentation 모델
results = model('image.jpg')

for result in results:
    masks = result.masks  # 픽셀 마스크
    for mask in masks:
        # mask.data: 이진 마스크
        # mask.xy: 마스크 윤곽선
        pass
```

### 4. 포즈 추정 (Pose Estimation)

**기능**: 사람의 관절점 탐지

**활용**:
- 자세 분석
- 운동 동작 평가
- 낙상 감지

**코드**:
```python
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')
results = model('person.jpg')

for result in results:
    keypoints = result.keypoints  # 17개 관절점
    for kp in keypoints:
        # kp.xy: (x, y) 좌표
        # kp.conf: 신뢰도
        pass
```

### 5. 이미지 분류 (Classification)

**기능**: 전체 이미지 클래스 분류

**활용**:
- 장면 인식
- 품질 검사
- 분류 작업

**코드**:
```python
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
results = model('image.jpg')

for result in results:
    top5 = result.probs.top5  # Top 5 클래스
    top5conf = result.probs.top5conf  # Top 5 신뢰도
```

### 6. 지향 경계 상자 (Oriented Bounding Box - OBB)

**기능**: 회전된 객체 탐지

**활용**:
- 항공 이미지 분석
- 문서 내 텍스트 탐지
- 회전된 물체 탐지

**코드**:
```python
from ultralytics import YOLO

model = YOLO('yolov8n-obb.pt')
results = model('aerial.jpg')

for result in results:
    obb = result.obb  # 회전 박스
```

### 7. 커스텀 객체 탐지

**기능**: 특정 도메인 객체 학습

**활용**:
- 제조 결함 탐지
- 특정 제품 인식
- 맞춤형 애플리케이션

**학습 코드**:
```python
from ultralytics import YOLO

# 데이터셋 준비 (YOLO 형식)
# dataset.yaml 작성

model = YOLO('yolov8n.pt')
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu'  # 라즈베리파이에서는 CPU
)
```

### 8. 실시간 영상 분석

**기능**:
- 객체 카운팅 (라인 통과)
- 체류 시간 측정
- 혼잡도 분석
- 이상 행동 탐지

**라인 카운팅 예시**:
```python
class LineCounter:
    def __init__(self, line_position):
        self.line = line_position  # (x1, y1, x2, y2)
        self.counted_ids = set()
        self.count = 0

    def check_crossing(self, track_id, center, prev_center):
        if track_id in self.counted_ids:
            return False

        # 라인 교차 확인
        if self.line_intersect(prev_center, center):
            self.counted_ids.add(track_id)
            self.count += 1
            return True
        return False

    def line_intersect(self, p1, p2):
        # 라인 교차 알고리즘
        pass
```

### 9. 영역 기반 탐지 (ROI Detection)

**기능**: 특정 영역 내 객체만 탐지

**활용**:
- 주차 공간 모니터링
- 금지 구역 침입 감지
- 영역별 통계

**코드**:
```python
import cv2
import numpy as np

def detect_in_roi(results, roi_points):
    roi_polygon = np.array(roi_points)
    detected_in_roi = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = ((x1+x2)//2, (y1+y2)//2)

        if cv2.pointPolygonTest(roi_polygon, center, False) >= 0:
            detected_in_roi.append(box)

    return detected_in_roi
```

### 10. 알림 및 자동화

**기능**: 탐지 이벤트 기반 액션

**활용**:
- 텔레그램/이메일 알림
- 자동 녹화
- IoT 연동 (조명, 경보)

**텔레그램 알림**:
```python
import requests

def send_telegram_alert(bot_token, chat_id, message, image_path=None):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}

    requests.post(url, data=data)

    if image_path:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        files = {'photo': open(image_path, 'rb')}
        data = {"chat_id": chat_id, "caption": message}
        requests.post(url, data=data, files=files)
```

---

## 구현된 활용 코드

이 프로젝트에는 산업별 활용을 위한 실용적인 코드가 구현되어 있습니다.

### 1. 침입 감지 시스템 (IntrusionDetector)

**위치**: `src/detection/intrusion_detector.py`

**기능**:
- ROI(Region of Interest) 기반 침입 감지
- 특정 영역 내 객체 진입 시 자동 알림
- 쿨다운 시스템으로 알림 스팸 방지
- 침입 통계 추적

**사용 예시**:
```python
from src.detection import IntrusionDetector

# ROI 정의 (다각형 좌표)
roi = [(200, 150), (440, 150), (440, 330), (200, 330)]

# 침입 감지기 초기화
detector = IntrusionDetector(
    roi_points=roi,
    target_classes=[0],  # 0 = person
    cooldown_seconds=5
)

# 탐지 결과로 침입 체크
intrusion, intruders = detector.check_intrusion(detections)

if intrusion and detector.should_alert():
    detector.trigger_alert()
    print(f"침입 감지! {len(intruders)}명")
```

**실행 방법**:
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

**활용 분야**: 보안, 감시, 출입 통제

---

### 2. 라인 통과 카운팅 (LineCrossingCounter)

**위치**: `src/detection/line_counter.py`

**기능**:
- 가상 라인 통과 카운팅
- 양방향 카운팅 지원 (IN/OUT)
- 객체 추적 기반 정확한 카운팅
- 중복 카운팅 방지

**사용 예시**:
```python
from src.detection import LineCrossingCounter

# 카운팅 라인 정의
line_start = (0, 240)
line_end = (640, 240)

# 카운터 초기화
counter = LineCrossingCounter(
    line_start=line_start,
    line_end=line_end,
    target_classes=[0],  # person only
    bidirectional=True
)

# 추적 정보를 포함한 탐지 결과로 업데이트
count_in, count_out, crossed = counter.update(detections)

print(f"IN: {count_in}, OUT: {count_out}")
```

**실행 방법**:
```bash
# 기본 실행
python examples/people_counting.py --camera usb

# 정기 리포트 전송 (60초마다)
python examples/people_counting.py \
  --camera usb \
  --report-interval 60 \
  --telegram-token YOUR_TOKEN \
  --telegram-chat-id YOUR_CHAT_ID
```

**활용 분야**: 리테일, 스마트시티, 교통 분석

---

### 3. 주차 공간 모니터링 (ParkingSpaceMonitor)

**위치**: `src/detection/specialized_detectors.py`

**기능**:
- 다중 주차 공간 점유 상태 추적
- 실시간 가용 공간 카운팅
- 예약석 관리
- 불법 주차 감지

**사용 예시**:
```python
from src.detection import ParkingSpaceMonitor

# 주차 공간 정의
parking_spaces = [
    {
        'id': 1,
        'points': [(50, 200), (150, 200), (150, 350), (50, 350)],
        'reserved': False
    },
    # ... 추가 공간
]

# 모니터 초기화
monitor = ParkingSpaceMonitor(parking_spaces)

# 차량 탐지 결과로 점유 상태 확인
space_status = monitor.check_occupancy(vehicles)

# 통계 조회
stats = monitor.get_statistics()
print(f"가용 공간: {stats['available']}개")
```

**실행 방법**:
```bash
# 기본 실행 (내장 설정)
python examples/parking_monitor.py --camera usb

# 설정 파일 사용
python examples/parking_monitor.py \
  --camera rtsp \
  --url rtsp://camera-ip/stream \
  --config configs/parking_config.yaml
```

**활용 분야**: 스마트시티, 주차장 관리

---

### 4. 안전장비 착용 확인 (SafetyEquipmentDetector)

**위치**: `src/detection/specialized_detectors.py`

**기능**:
- 헬멧, 안전조끼, 안전화 착용 확인
- 사람과 장비 매칭 알고리즘
- 위반자 자동 감지
- 준수율 통계

**사용 예시**:
```python
from src.detection import SafetyEquipmentDetector

# 필수 장비 정의
detector = SafetyEquipmentDetector(
    required_equipment=['helmet', 'vest'],
    check_distance_threshold=100
)

# 사람과 장비 탐지 결과 분리
persons = [d for d in detections if d['class_name'] == 'person']
equipment = [d for d in detections if d['class_name'] in ['helmet', 'vest']]

# 준수 여부 확인
compliant, violations = detector.check_compliance(persons, equipment)

print(f"위반자: {len(violations)}명")
for person in violations:
    print(f"  - 누락 장비: {', '.join(person['missing'])}")
```

**활용 분야**: 제조, 건설, 산업 안전

---

### 5. 알림 시스템 (TelegramNotifier, EmailNotifier)

**위치**: `src/utils/notifier.py`

**기능**:
- 텔레그램 봇을 통한 실시간 알림
- 이메일 알림 (이미지 첨부 지원)
- 다중 채널 통합 관리
- 알림 템플릿 자동 생성

**텔레그램 사용 예시**:
```python
from src.utils import TelegramNotifier

# 알림 초기화
notifier = TelegramNotifier(
    bot_token='YOUR_BOT_TOKEN',
    chat_id='YOUR_CHAT_ID'
)

# 간단한 메시지
notifier.send_message("시스템 시작됨")

# 이미지와 함께 알림
notifier.send_alert(
    alert_type='intrusion',
    message='침입자 탐지!\n위치: 정문\n시간: 14:30',
    image_path='alert.jpg'
)
```

**이메일 사용 예시**:
```python
from src.utils import EmailNotifier

# 이메일 알림 초기화
notifier = EmailNotifier(
    smtp_server='smtp.gmail.com',
    smtp_port=587,
    sender_email='your@gmail.com',
    sender_password='app_password',
    recipient_email='recipient@example.com'
)

# 알림 전송
notifier.send_alert(
    alert_type='violation',
    message='안전장비 미착용 감지',
    image_path='violation.jpg'
)
```

**다중 채널 사용**:
```python
from src.utils import MultiNotifier

# 모든 채널 통합
notifier = MultiNotifier()
notifier.add_telegram(bot_token, chat_id)
notifier.add_email(smtp_server, smtp_port, sender, password, recipient)

# 모든 채널로 동시 전송
notifier.send_alert('alert', '중요 이벤트 발생!')
```

**환경 설정**:
```bash
# .env 파일 생성
cp .env.example .env

# 필수 값 설정
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

**활용 분야**: 모든 산업 (범용)

---

### 실용 예제 스크립트

#### 침입 감지 시스템
```bash
examples/intrusion_detection.py
```
- ROI 기반 침입 감지
- 실시간 알림
- 통계 추적

#### 사람 카운팅
```bash
examples/people_counting.py
```
- 양방향 카운팅
- 정기 리포트
- 통계 대시보드

#### 주차장 모니터링
```bash
examples/parking_monitor.py
```
- 다중 공간 추적
- 가용 공간 표시
- 점유율 분석

#### 혼잡도 분석
```bash
examples/crowd_density.py
```
- 다중 영역 밀도 분석
- 실시간 혼잡도 레벨
- 영역별 통계

#### 이벤트 기반 녹화
```bash
examples/event_recorder.py
```
- 자동 비디오 녹화
- 이벤트 전후 버퍼링
- 파일 크기 관리

#### 고급 객체 추적
```bash
examples/object_tracker.py
```
- 이동 경로 추적
- 체류 시간 측정
- 히트맵 생성

#### 낙상 감지 시스템
```bash
examples/fall_detection.py
```
- YOLOv8-pose 기반 포즈 추정
- 실시간 낙상 감지
- 자동 녹화 및 알림
- 포즈 타입 분류

---

## 산업별 활용 사례

### 1. 보안 및 감시

**구현 코드**: `examples/intrusion_detection.py`

**용도**:
- 침입 감지 시스템
- 무단 출입 모니터링
- 실종자 찾기
- 군중 통제

**구현 예시**:
- 특정 영역에 사람 진입 시 알림
- 허가되지 않은 시간대 움직임 탐지
- 얼굴 마스킹 준수 확인

**시장 규모**: 전 세계 비디오 감시 시장 $62B (2023)

### 2. 스마트 시티

**구현 코드**: `examples/people_counting.py`, `examples/parking_monitor.py`

**용도**:
- 교통량 분석
- 주차 공간 관리
- 쓰레기통 만차 감지
- 가로등 제어

**구현 예시**:
- 실시간 교통 혼잡도 측정 (LineCrossingCounter)
- 불법 주차 탐지 (ParkingSpaceMonitor)
- 보행자 안전 모니터링

**실제 구현 기능**:
```python
# 교통량 카운팅
counter = LineCrossingCounter(
    line_start=(0, 300),
    line_end=(640, 300),
    target_classes=[2, 3, 5, 7]  # 차량 클래스
)

# 주차 공간 모니터링
monitor = ParkingSpaceMonitor(parking_spaces)
stats = monitor.get_statistics()
print(f"가용 주차 공간: {stats['available']}개")
```

**관련 프로젝트**:
- 서울시 스마트시티 (2024)
- 싱가포르 Smart Nation

### 3. 제조 및 품질 관리

**구현 코드**: `src/detection/specialized_detectors.py` (SafetyEquipmentDetector)

**용도**:
- 제품 결함 탐지
- 조립 라인 모니터링
- 안전장비 착용 확인
- 재고 관리

**구현 예시**:
- PCB 결함 검사 (커스텀 모델 학습 필요)
- 헬멧/안전복 미착용자 감지 (SafetyEquipmentDetector)
- 부품 누락 확인

**실제 구현 기능**:
```python
# 안전장비 착용 확인
detector = SafetyEquipmentDetector(
    required_equipment=['helmet', 'vest'],
    check_distance_threshold=100
)

compliant, violations = detector.check_compliance(persons, equipment)

# 위반자 알림
if violations:
    for person in violations:
        print(f"위반: {person['missing']}")
        notifier.send_alert('violation', f"안전장비 미착용 감지")
```

**정확도**: 99%+ (맞춤 학습 모델)

### 4. 농업 (스마트팜)

**용도**:
- 작물 병해충 탐지
- 수확 시기 판단
- 가축 행동 모니터링
- 과일 숙성도 분석

**구현 예시**:
- 잎 질병 분류
- 닭 개체수 카운팅
- 소 발정기 탐지

**시장**: 정밀 농업 시장 $12B (2025 예상)

### 5. 리테일 (소매업)

**구현 코드**: `examples/people_counting.py`, `src/detection/intrusion_detector.py`

**용도**:
- 고객 동선 분석
- 재고 모니터링
- 도난 방지
- 셀프 계산대

**구현 예시**:
- 고객 입장/퇴장 카운팅 (LineCrossingCounter)
- 제한 구역 침입 감지 (IntrusionDetector)
- 매장 혼잡도 분석

**실제 구현 기능**:
```python
# 고객 카운팅 (입장/퇴장)
counter = LineCrossingCounter(
    line_start=(320, 0),
    line_end=(320, 480),
    target_classes=[0],  # person
    bidirectional=True
)

# 시간대별 통계 수집
stats = counter.get_statistics()
print(f"총 방문자: {stats['total']}명")
```

**효과**: 30% 도난 감소, 20% 재고 효율 증가

### 6. 의료 및 헬스케어

**용도**:
- 낙상 감지
- 환자 모니터링
- 의료 기구 추적
- 위생 규정 준수

**구현 예시**:
- 노인 낙상 자동 알림
- 마스크 착용 확인
- 손 씻기 모니터링

**적용**: 요양원, 병원, 격리 시설

### 7. 물류 및 창고

**용도**:
- 패키지 분류
- 팔레트 추적
- 안전 관리
- 재고 카운팅

**구현 예시**:
- 바코드/QR 인식
- 지게차 안전 거리 모니터링
- 자동 재고 조사

**효과**: 50% 재고 조사 시간 단축

### 8. 교통 및 운송

**용도**:
- 번호판 인식 (ANPR)
- 교통 위반 탐지
- 사고 감지
- 주차 관리

**구현 예시**:
- 신호 위반 자동 촬영
- 역주행 탐지
- 보행자 우선 신호 제어

**전 세계 도입**: 100+ 도시

### 9. 환경 보호

**용도**:
- 야생동물 모니터링
- 불법 벌목 감지
- 쓰레기 분류
- 산불 조기 탐지

**구현 예시**:
- 멸종 위기 동물 개체수 조사
- 밀렵꾼 탐지
- 재활용 쓰레기 자동 분류

**사례**: Amazon 열대우림 모니터링

### 10. 스포츠 및 피트니스

**용도**:
- 자세 교정
- 동작 분석
- 선수 추적
- 퍼포먼스 분석

**구현 예시**:
- 스쿼트 자세 평가
- 골프 스윙 분석
- 축구 선수 위치 추적

**활용**: 프로 스포츠팀, 피트니스 앱

### 산업별 YOLO 모델 선택

| 산업 | 권장 모델 | 우선 기능 |
|------|----------|----------|
| 보안/감시 | YOLOv8n + Tracking | 실시간성 |
| 제조 | YOLOv8s + Custom | 정확도 |
| 농업 | YOLOv8n-seg | 세그멘테이션 |
| 의료 | YOLOv8n-pose | 포즈 추정 |
| 리테일 | YOLOv8n + Tracking | 카운팅 |

---

## 성능 최적화

### 라즈베리파이 최적화 팁

#### 1. 모델 경량화

```python
# INT8 양자화 (TFLite)
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='tflite', int8=True)

# TFLite 추론
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='yolov8n_int8.tflite')
interpreter.allocate_tensors()
```

#### 2. 해상도 조정

```python
# 입력 크기 감소
model = YOLO('yolov8n.pt')
results = model(frame, imgsz=320)  # 640 -> 320
```

#### 3. 프레임 스킵

```python
frame_counter = 0
process_every_n_frames = 3

while True:
    ret, frame = cap.read()
    frame_counter += 1

    if frame_counter % process_every_n_frames == 0:
        results = model(frame)
        # 처리

    # 항상 표시
    cv2.imshow('Frame', frame)
```

#### 4. ROI 처리

```python
# 전체 프레임 대신 관심 영역만 처리
roi = frame[100:400, 200:600]  # y1:y2, x1:x2
results = model(roi)
```

#### 5. 신뢰도 임계값 조정

```python
# 낮은 신뢰도 검출 제외로 후처리 시간 단축
results = model(frame, conf=0.6)  # 기본 0.25
```

#### 6. GPU 가속 (Raspberry Pi 4/5)

```bash
# OpenCV with OpenGL ES 지원
sudo apt install libgles2-mesa-dev

# PyTorch with ARM NEON
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

#### 7. 시스템 설정

```bash
# GPU 메모리 증가 (CSI 카메라용)
sudo raspi-config
# Performance Options > GPU Memory > 256

# CPU 오버클럭 (자체 책임)
sudo nano /boot/config.txt
# over_voltage=6
# arm_freq=2000

# 불필요한 서비스 비활성화
sudo systemctl disable bluetooth
sudo systemctl disable wifi
```

#### 8. 성능 벤치마크

```python
import time

def benchmark_model(model, frame, iterations=100):
    times = []
    for _ in range(iterations):
        start = time.time()
        results = model(frame, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    fps = 1 / avg_time
    print(f"평균 추론 시간: {avg_time*1000:.2f}ms")
    print(f"예상 FPS: {fps:.2f}")
    return avg_time, fps
```

### 예상 성능 (Raspberry Pi 4 4GB)

| 모델 | 입력 크기 | FPS | 정확도 (mAP) |
|------|----------|-----|--------------|
| YOLOv5n | 320x320 | 8-10 | 28.0 |
| YOLOv5n | 640x640 | 4-6 | 28.0 |
| YOLOv8n | 320x320 | 6-8 | 37.3 |
| YOLOv8n | 640x640 | 3-5 | 37.3 |
| YOLOv8s | 640x640 | 1-2 | 44.9 |

---

## 프로젝트 구조 예시

```
rasp_vision_yolo/
├── configs/
│   ├── camera_config.yaml
│   ├── model_config.yaml
│   └── detection_zones.yaml
├── models/
│   ├── yolov8n.pt
│   ├── yolov8n.onnx
│   └── custom_model.pt
├── src/
│   ├── camera/
│   │   ├── __init__.py
│   │   ├── csi_camera.py
│   │   ├── usb_camera.py
│   │   └── rtsp_camera.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py
│   │   ├── tracker.py
│   │   └── roi_handler.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── frame_processor.py
│   │   └── post_processor.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── notifier.py
│   │   └── visualizer.py
│   └── main.py
├── tests/
│   ├── test_camera.py
│   ├── test_detection.py
│   └── test_integration.py
├── logs/
├── recordings/
├── requirements.txt
├── setup.sh
└── README.md
```

---

## 다음 단계

1. **요구사항 정의**: 구체적인 사용 사례 결정
2. **하드웨어 준비**: 라즈베리파이, 카메라, 전원 준비
3. **환경 설정**: OS 설치 및 패키지 설치
4. **프로토타입**: 기본 탐지 기능 구현
5. **최적화**: 성능 튜닝
6. **배포**: 실제 환경 적용

## 추가 리소스

- **Ultralytics 공식 문서**: https://docs.ultralytics.com
- **YOLOv5 GitHub**: https://github.com/ultralytics/yolov5
- **Raspberry Pi 카메라 가이드**: https://www.raspberrypi.com/documentation/accessories/camera.html
- **OpenCV 튜토리얼**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

---

**작성일**: 2025-11-12
**버전**: 1.0
