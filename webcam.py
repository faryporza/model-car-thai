import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def main():
    # Load YOLOv11s model
    model_path = Path(__file__).parent / 'best.pt'
    model = YOLO(model_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run inference
        results = model(frame)
        
        # Render results on frame
        annotated_frame = results[0].plot()
        
        # Display frame
        cv2.imshow('YOLOv11s Webcam Detection', annotated_frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
