from ultralytics import YOLO
import numpy as np


class Detector:
    def __init__(self, model_path="yolov8m-football.pt", conf_threshold=0.1):
        self.name = "Detector"
        self.model_path = model_path
        self.conf_threshold = conf_threshold

    def start(self, data):
        print("[INFO] YOLOv8 Detector wurde gestartet.")
        self.model = YOLO(self.model_path)

    def stop(self, data):
        print("[INFO] YOLOv8 Detector wurde gestoppt.")

    def step(self, data):
        image = data["image"]
        results = self.model.predict(image, verbose=False)
        boxes = results[0].boxes

        detections = []
        classes = []

        for box in boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if conf < self.conf_threshold:
                continue

            if cls_id in [0, 1, 2, 3]:
                xywh = box.xywh[0].cpu().numpy()
                detections.append(xywh)
                classes.append(cls_id)
            else:
                continue

        if detections:
            detections = np.stack(detections)
            classes = np.array(classes).reshape(-1)
        else:
            detections = np.zeros((0, 4))
            classes = np.zeros((0,), dtype=int)
        results = {"detections": detections, "classes": classes}
        return results

        # TODO: Implement processing of a single frame
        # The task of the detector is to detect the ball, the goal keepers, the players and the referees if visible.
        # A bounding box needs to be defined for each detected object including the objects center position (X,Y) and its width and height (W, H)
        # You can return an arbitrary number of objects

        # Note: You can access data["image"] to receive the current image
        # Return a dictionary with detections and classes
        #
        # Detections must be a Nx4 NumPy Tensor, one 4-dimensional vector per detection
        # The detection vector itself is encoded as (X, Y, W, H), so X and Y coordinate first, then width and height of each detection box.
        # X and Y coordinates are the center point of the object, so the bounding box is drawn from (X - W/2, Y - H/2) to (X + W/2, Y + H/2)
        #
        # Classes must be Nx1 NumPy Tensor, one scalar entryx per detection
        # For each corresponding detection, the following mapping must be used
        #   0: Ball
        #   1: GoalKeeper
        #   2: Player
        #   3: Referee
