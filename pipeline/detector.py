from ultralytics import YOLO
import torch
import numpy as np
import os

class Detector:
    def __init__(self):
        self.name = "Detector"
        self.model = None

    def initialise_yolo(self):
        model_path = "/Users/jule/Documents/Uni/4. Semester/Machine Perception und Tracking/mpt_football_aufgabenstellung/yolov8n-football.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please download it.")
        self.model = YOLO(model_path)

    def start(self, data):
        if self.model is None:
            self.initialise_yolo()

    def stop(self, data):
        self.model = None

    def step(self, data):
        image = data["image"]
        if self.model is None:
            self.initialise_yolo()

        results = self.model(image)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return {
            "detections": np.empty((0, 4), dtype=np.float32),
            "classes": np.empty((0, 1), dtype=np.int64)
            }

        boxes = results.boxes.xywh.cpu().numpy()   # Center-based boxes: (X, Y, W, H)
        classes = results.boxes.cls.cpu().numpy()  # Klassen

        valid_classes = [0, 1, 2, 3]
        indices = [i for i, cls in enumerate(classes) if int(cls) in valid_classes]

        if not indices:
            return {
            "detections": np.empty((0, 4), dtype=np.float32),
            "classes": np.empty((0, 1), dtype=np.int64)
            }

        detections = np.array([boxes[i] for i in indices], dtype=np.float32)
        class_tensor = np.array([[int(classes[i])] for i in indices], dtype=np.int64)

        return {
        "detections": detections,
        "classes": class_tensor
        }
