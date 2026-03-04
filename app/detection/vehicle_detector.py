from ultralytics import YOLO
import torch

# COCO class IDs for vehicles only
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]

class VehicleDetector:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            # tracker="bytetrack.yaml",
            tracker="botsort.yaml",
            conf=0.4,
            iou=0.5,
            classes=VEHICLE_CLASSES,   # ← only cars, motorcycles, buses, trucks
            verbose=False
        )
        return results
