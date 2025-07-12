# model-car-thai

### 1. สร้าง Virtual Environment (แนะนำ)

```bash
# สร้าง virtual environment
python3 -m venv yolo_env

# เปิดใช้งาน virtual environment
# บน Windows:
yolo_env\Scripts\activate
# บน macOS/Linux:
source yolo_env/bin/activate
```

### 2. ติดตั้ง package

```bash
pip install opencv-python ultralytics roboflow supervision pyyaml torch
```

### 3. เทรนโมเดล

```bash
python train.py
```

### 4. ทดสอบด้วย webcam

```bash
python webcam.py
```

### 5. โครงสร้างโปรเจกต์

```
MDAll/
├── train.py          # สำหรับเทรนโมเดล
├── webcam.py         # สำหรับทดสอบด้วย webcam
├── data.yaml         # ไฟล์ config สำหรับ dataset
├── runs/             # ผลลัพธ์การเทรนและ inference
│   ├── detect/
│   │   ├── train/    # ผลลัพธ์การเทรน
│   │   └── predict/  # ผลลัพธ์ inference
└── README.md
```

### 6. หมายเหตุ

- โมเดลที่เทรนแล้วจะถูกบันทึกไว้ที่ `runs/detect/train/weights/best.pt`
- ใช้ GPU ถ้ามี (CUDA) สำหรับเทรนเร็วขึ้น
- สามารถปรับ epochs, imgsz, และ model size ใน `train.py` ได้
- นะจ๊ะ
