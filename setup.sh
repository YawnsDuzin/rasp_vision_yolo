#!/bin/bash

# Raspberry Pi YOLO Vision Project Setup Script

set -e

echo "========================================="
echo "Raspberry Pi YOLO Vision Setup"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo -e "${YELLOW}Warning: This script is designed for Raspberry Pi${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}Step 1: System Update${NC}"
sudo apt update
sudo apt upgrade -y

echo -e "${GREEN}Step 2: Installing System Dependencies${NC}"
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    git \
    cmake

echo -e "${GREEN}Step 3: Camera Support${NC}"
read -p "Install CSI camera support (picamera2)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo apt install -y python3-picamera2
    echo -e "${GREEN}picamera2 installed${NC}"
fi

echo -e "${GREEN}Step 4: Python Virtual Environment${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

echo -e "${GREEN}Step 5: Activating Virtual Environment${NC}"
source venv/bin/activate

echo -e "${GREEN}Step 6: Upgrading pip${NC}"
pip install --upgrade pip

echo -e "${GREEN}Step 7: Installing Python Dependencies${NC}"
pip install -r requirements.txt

echo -e "${GREEN}Step 8: Creating Required Directories${NC}"
mkdir -p models logs recordings

# Add .gitkeep files
touch models/.gitkeep
touch logs/.gitkeep
touch recordings/.gitkeep

echo -e "${GREEN}Step 9: Downloading YOLO Model${NC}"
read -p "Download YOLOv8n model (~6MB)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model downloaded successfully')"
    if [ -f "yolov8n.pt" ]; then
        mv yolov8n.pt models/
        echo -e "${GREEN}YOLOv8n model downloaded to models/${NC}"
    fi
fi

echo -e "${GREEN}Step 10: Permissions${NC}"
# Make scripts executable
chmod +x setup.sh
chmod +x examples/*.py 2>/dev/null || true

echo ""
echo "========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Test camera:"
echo "   python examples/camera_test.py --camera usb"
echo ""
echo "3. Run YOLO detection:"
echo "   python examples/yolo_realtime.py --camera usb"
echo ""
echo "4. Read documentation:"
echo "   cat DEVELOPMENT_GUIDE.md"
echo ""
echo -e "${GREEN}Happy coding!${NC}"
