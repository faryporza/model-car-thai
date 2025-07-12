import subprocess
import sys

# Install dependencies
def install_packages():
    packages = ["ultralytics", "supervision", "roboflow", "pyyaml", "torch"]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Run installation
install_packages()

from roboflow import Roboflow
from ultralytics import YOLO
import yaml
import os
import glob
import torch

# Setup API and download dataset
rf = Roboflow(api_key="xWmGZqldGGdVHAeTBz8b")
project = rf.workspace("thaidetec").project("vehicle-detection-yg4le")
version = project.version(5)
dataset = version.download("yolov11")  # Gets path of data.yaml and image folders

# Modify data.yaml to include test folder
with open(os.path.join(dataset.location, "data.yaml")) as f:
    data = yaml.safe_load(f)

# Add test folder if it doesn't exist
data["train"] = os.path.join(dataset.location, "train", "images")
data["val"]   = os.path.join(dataset.location, "valid", "images")
data["test"]  = os.path.join(dataset.location, "test", "images")

with open("data.yaml", "w") as f:
    yaml.dump(data, f)

# Start training YOLOv11s
model = YOLO("yolo11m.pt")  # Can change to yolo11n/m/l as needed
results = model.train(
    task="detect",
    data="data.yaml",
    epochs=100,
    imgsz=640,
    plots=True,     # plot training curves
    device="0" if torch.cuda.is_available() else "cpu"
)

# Display confusion matrix (if running in Jupyter)
try:
    from IPython.display import Image
    Image(filename="runs/detect/train/confusion_matrix.png")
except ImportError:
    print("Confusion matrix saved to: runs/detect/train/confusion_matrix.png")

# Run inference to see results
model_path = "runs/detect/train/weights/best.pt"
test_images = os.path.join(dataset.location, "test", "images")

# Load trained model and run prediction
trained_model = YOLO(model_path)
results = trained_model.predict(source=test_images, save=True, conf=0.25)

# Display first 3 result images from runs/detect/predict folder
try:
    latest = max(glob.glob('runs/detect/predict*'), key=os.path.getmtime)
    for img in glob.glob(f"{latest}/*.jpg")[:3]:
        from IPython.display import Image as IPyImage, display
        display(IPyImage(filename=img, width=600))
except ImportError:
    print("Results saved to: runs/detect/predict/")
    print("Check the folder for inference results")
