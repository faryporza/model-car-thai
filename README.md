# model-car-thai

### 1. สร้าง Virtual Environment (แนะนำ)

```bash
# สร้าง virtual environment
python3.10.18 -m venv yolo_env

# เปิดใช้งาน virtual environment
# บน Windows:
yolo_env\Scripts\activate
# บน macOS/Linux:
source yolo_env/bin/activate
```

### 2. ติดตั้ง package

```bash
pip install opencv-python ultralytics
```

### 3. run เพื่อทดสอบ

```bash
python webcam.py
```