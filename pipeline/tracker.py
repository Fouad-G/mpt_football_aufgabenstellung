import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(boxes1, boxes2):
    """
    Computes the Intersection over Union (IoU) between two sets of bounding boxes.

    Parameters:
        boxes1 (np.ndarray): First set of bounding boxes, shape (N, 4)
        boxes2 (np.ndarray): Second set of bounding boxes, shape (M, 4)

    Returns:
        np.ndarray: IoU matrix of shape (N, M)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    # (X, Y, W, H) → (x1, y1) + (x2, y2)
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    a1 = boxes1[:, :2] - boxes1[:, 2:] / 2
    a2 = boxes1[:, :2] + boxes1[:, 2:] / 2
    b1 = boxes2[:, :2] - boxes2[:, 2:] / 2
    b2 = boxes2[:, :2] + boxes2[:, 2:] / 2

    tl = np.maximum(a1[:, None], b1[None])
    br = np.minimum(a2[:, None], b2[None])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = boxes1[:, 2] * boxes1[:, 3]
    area_b = boxes2[:, 2] * boxes2[:, 3]
    union = area_a[:, None] + area_b[None] - inter

    return inter / np.clip(union, 1e-6, None)


class Filter:
    """
    Represents a single track (filter) with state, velocity, and class.
    """

    _next_id = 0  # Global unique ID counter

    def __init__(self, bbox, obj_class):
        self.state = bbox.astype(float)  # [cx, cy, w, h]
        self.velocity = np.zeros(2)  # [vx, vy]
        self.cls = int(obj_class)  # Class ID
        self.age = 1
        self.missing = 0
        self.id = Filter._next_id
        Filter._next_id += 1

    def predict(self, flow):
        """
        Predicts the new state based on velocity and optical flow.
        """
        self.state[:2] += self.velocity - flow
        self.age += 1
        self.missing += 1

    def update(self, bbox):
        """
        Updates the track with a new detection (bounding box).
        """
        alpha = 0.6
        beta = 0.15
        innovation = bbox[:2] - self.state[:2]
        self.state[:2] += alpha * innovation
        self.velocity = (1 - beta) * self.velocity + beta * innovation
        self.state[2:] = bbox[2:]
        self.missing = 0

    def to_bbox(self):
        """
        Returns the current state as bounding box [cx, cy, w, h].
        """
        return self.state.copy()


class Tracker:
    """
    Tracks objects across frames using IoU and a simple motion model.
    """

    def __init__(self, iou_threshold=0.3):
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_missing = {0: 1, 1: 5, 2: 5, 3: 5}
        self.vmax = {0: 120, 1: 50, 2: 50, 3: 50}
        self.name = "Tracker"

    def start(self, data):
        """
        Called once at module startup.
        """
        self.tracks = []

    def stop(self, data):
        """
        Called once at module shutdown.
        """
        pass

    def step(self, data):
        """
        Executes tracking step on the current frame.

        Parameters:
            data (dict): Input dictionary containing:
                - "detections": np.ndarray of bounding boxes
                - "classes": list of class IDs
                - "opticalFlow": 2D flow vector
                - "image": current frame image

        Returns:
            dict: Tracking results with positions, velocities, ages, classes, and IDs.
        """
        detections = data.get("detections", np.empty((0, 4)))
        classes = data.get("classes", np.empty((0, 1)))
        flow = data.get("opticalFlow", np.zeros(2))
        image = data.get("image", np.zeros((1080, 1920, 3)))
        h, w = image.shape[:2]

        # 1. Predict tracks
        for tr in self.tracks:
            tr.predict(flow)

        # 2. Compute cost matrix (based on IoU and distance)
        predicted = np.array([tr.to_bbox() for tr in self.tracks])
        cost = np.ones((len(self.tracks), len(detections)))
        if len(predicted) > 0 and len(detections) > 0:
            ious = iou_matrix(predicted, detections)
            dist = np.linalg.norm(
                predicted[:, None, :2] - detections[None, :, :2], axis=2
            )
            mask = dist < 160
            cost[mask] = 1 - ious[mask]

        # 3. Match tracks to detections using Hungarian algorithm
        matches = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))
        if cost.size:
            r, c = linear_sum_assignment(cost)
            for tr_idx, det_idx in zip(r, c):
                if (
                    cost[tr_idx, det_idx] < (1 - self.iou_threshold)
                    and self.tracks[tr_idx].cls == classes[det_idx]
                ):
                    matches.append((tr_idx, det_idx))
                    unmatched_tracks.discard(tr_idx)
                    unmatched_dets.discard(det_idx)

        # 4. Update matched tracks
        for tr_idx, det_idx in matches:
            self.tracks[tr_idx].update(detections[det_idx])

        # 5. Create new tracks for unmatched detections
        for di in unmatched_dets:
            if self.tracks:
                best_iou = iou_matrix(
                    detections[di : di + 1],
                    np.array([tr.to_bbox() for tr in self.tracks]),
                ).max()
                if best_iou > 0.45:
                    continue
            self.tracks.append(Filter(detections[di], classes[di]))

        # 6. Remove old or invalid tracks
        alive = []
        for tr in self.tracks:
            vmax = self.vmax.get(tr.cls, 50)
            tr.velocity = np.clip(tr.velocity, -vmax, vmax)
            if not self._is_on_field(tr, w, h):
                continue
            if tr.missing > self.max_missing.get(tr.cls, 5):
                continue
            alive.append(tr)
        self.tracks = alive

        # 7. Output results
        return {
            "tracks": np.array([tr.to_bbox() for tr in self.tracks]),
            "trackVelocities": np.array([tr.velocity for tr in self.tracks]),
            "trackAge": [tr.age for tr in self.tracks],
            "trackClasses": [tr.cls for tr in self.tracks],
            "trackIds": [tr.id for tr in self.tracks],
        }

    def _is_on_field(self, tr, w, h):
        """
        Checks if the track is still within the image boundaries and size is valid.

        Parameters:
            tr (Filter): Track object
            w (int): Image width
            h (int): Image height

        Returns:
            bool: True if the track is valid, False otherwise.
        """
        cx, cy, bw, bh = tr.to_bbox()
        return 0 <= cx <= w and 0 <= cy <= h and bw >= 2 and bh >= 2
