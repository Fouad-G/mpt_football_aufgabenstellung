from ultralytics import YOLO
import torch
import numpy as np
import os


class Detector:
    def __init__(self):
        """Initialize the Detector class.
        This class is responsible for loading the YOLO model and processing images to detect objects.
        """
        self.name = "Detector"
        self.model = None

    def initialise_yolo(self):
        """Load the YOLO model from a specified path.
        Raises:
            FileNotFoundError: If the model file does not exist at the specified path.
        """
        # Automatische Modellwahl basierend auf Verfügbarkeit einer GPU
        model_path = (
            "yolov8m-football.pt"
            if torch.cuda.is_available()
            else "yolov8n-football.pt"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file {model_path} not found. Please download it."
            )
        self.model = YOLO(model_path)

    def start(self, data):
        """Start the Detector by initializing the YOLO model.
        Args:
            data (dict): Input data, not used in this method but required for compatibility with the pipeline.
        """
        if self.model is None:
            self.initialise_yolo()

    def stop(self, data):
        """Stop the Detector by clearing the model.
        Args:
            data (dict): Input data, not used in this method but required for compatibility with the pipeline.
        """
        self.model = None

    def step(self, data):
        """Process a single step of the pipeline.
        Args:
            data (dict): Input data containing the image to be processed.
        Returns:
            dict: A dictionary containing the detections and their corresponding classes.
        """
        image = data["image"]
        if self.model is None:
            self.initialise_yolo()

        results = self.model(image)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return {
                "detections": np.empty((0, 4), dtype=np.float32),
                "classes": np.empty((0, 1), dtype=np.int64),
            }

        boxes = results.boxes.xywh.cpu().numpy()  # Center-based boxes: (X, Y, W, H)
        classes = results.boxes.cls.cpu().numpy()  # Klassen

        valid_classes = [
            0,
            1,
            2,
            3,
        ]  # Valid classes: 0=Ball, 1=Player, 2=Referee, 3=Goalkeeper
        # Filter indices for valid classes
        indices = [i for i, cls in enumerate(classes) if int(cls) in valid_classes]

        # If no valid detections, return empty arrays
        if not indices:
            return {
                "detections": np.empty((0, 4), dtype=np.float32),
                "classes": np.empty((0, 1), dtype=np.int64),
            }

        # Extract the filtered detections and their corresponding classes
        detections = np.array([boxes[i] for i in indices], dtype=np.float32)
        class_tensor = np.array([[int(classes[i])] for i in indices], dtype=np.int64)

        return {"detections": detections, "classes": class_tensor}
      