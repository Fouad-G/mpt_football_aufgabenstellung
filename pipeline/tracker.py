import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    def __init__(
        self,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        embedder="mobilenet",
        embedder_gpu=True,
        half=False,
        bgr=True,
    ):
        self.name = "Tracker"
        self.deepsort = DeepSort(
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            max_cosine_distance=max_cosine_distance,
            embedder=embedder,
            embedder_gpu=embedder_gpu,
            half=half,
            bgr=bgr,
        )

        self.track_ages = {}
        self.track_classes = {}
        self.prev_positions = {}
        self.track_velocities = {}

    def start(self, data):
        print("[INFO] DeepSort Tracker wurde gestartet.")

    def stop(self, data):
        print("[INFO] DeepSort Tracker wurde gestoppt.")

    def _xywh_to_xyxy(self, x_center, y_center, w, h):
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return [x1, y1, x2, y2]

    def _xyxy_to_xywh(self, x1, y1, x2, y2):
        w = x2 - x1
        h = y2 - y1
        x_center = x1 + w / 2
        y_center = y1 + h / 2
        return [x_center, y_center, w, h]

    def compute_iou(self, boxA, boxB):
        # boxA und boxB im Format [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def step(self, data):
        detections = data.get("detections", [])
        classes = data.get("classes", [])
        frame = data.get("image", None)

        if frame is None:
            print("[WARN] Kein Frame übergeben, ReID nicht möglich!")
            return {
                "tracks": np.zeros((0, 4)),
                "trackVelocities": np.zeros((0, 2)),
                "trackAge": [],
                "trackClasses": [],
                "trackIds": [],
                "teamClasses": [],
            }
        if len(detections) == 0:
            return {
                "tracks": np.zeros((0, 4)),
                "trackVelocities": np.zeros((0, 2)),
                "trackAge": [],
                "trackClasses": [],
                "trackIds": [],
                "teamClasses": [],
            }

        raw_detections = []
        detection_boxes = []
        detection_classes = []
        for i, det in enumerate(detections):
            x_center, y_center, w, h = det
            x1, y1, x2, y2 = self._xywh_to_xyxy(x_center, y_center, w, h)
            left, top = x1, y1
            ww = x2 - x1
            hh = y2 - y1
            conf = 1.0
            cls_val = int(classes[i]) if i < len(classes) else 0
            raw_detections.append(([left, top, ww, hh], conf, cls_val))
            detection_boxes.append([x1, y1, x2, y2])
            detection_classes.append(cls_val)

        outputs = self.deepsort.update_tracks(raw_detections, frame=frame)

        tracked_positions = []
        tracked_velocities = []
        tracked_ages = []
        tracked_classes = []
        tracked_ids = []

        for t in outputs:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue

            track_id = t.track_id
            l, t_, r, b = t.to_tlbr()
            x_center, y_center, w_, h_ = self._xyxy_to_xywh(l, t_, r, b)

            if track_id not in self.track_ages:
                self.track_ages[track_id] = 1
            else:
                self.track_ages[track_id] += 1

            best_iou = 0.0
            best_cls = 0
            track_box = [l, t_, r, b]
            for i, det_box in enumerate(detection_boxes):
                iou = self.compute_iou(track_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_cls = detection_classes[i]
            if best_iou > 0.3:
                cls_in_tracker = best_cls
            else:
                cls_in_tracker = 0

            self.track_classes[track_id] = cls_in_tracker

            if track_id in self.prev_positions:
                px_center, py_center, _, _ = self.prev_positions[track_id]
                vx = x_center - px_center
                vy = y_center - py_center
                self.track_velocities[track_id] = (vx, vy)
            else:
                self.track_velocities[track_id] = (0.0, 0.0)

            self.prev_positions[track_id] = (x_center, y_center, w_, h_)

            tracked_positions.append([x_center, y_center, w_, h_])
            tracked_velocities.append(self.track_velocities[track_id])
            tracked_ages.append(self.track_ages[track_id])
            tracked_classes.append(cls_in_tracker)
            tracked_ids.append(track_id)

        team_classes = data.get("teamClasses", [])
        if len(team_classes) == 0:
            team_classes = [0] * len(tracked_positions)

        if len(tracked_positions) == 0:
            tracks = np.zeros((0, 4))
        else:
            tracks = np.array(tracked_positions).reshape(-1, 4)

        if len(tracked_velocities) == 0:
            track_velocities = np.zeros((0, 2))
        else:
            track_velocities = np.array(tracked_velocities).reshape(-1, 2)

        results = {
            "tracks": tracks,
            "trackVelocities": track_velocities,
            "trackAge": tracked_ages,
            "trackClasses": tracked_classes,
            "trackIds": tracked_ids,
            "teamClasses": team_classes,
        }

        return results

        # TODO: Implement processing of a detection list
        # The task of the tracker module is to identify (temporal) consistent tracks out of the given list of detections
        # The tracker maintains a list of known tracks which is initially empty.
        # The tracker then tries to associate all given detections from the detector to existing tracks. A meaningful metric needs to be defined
        # to decide which detection should be associated with each track and which detections better stay unassigned.
        # After the association step, one must handle there different cases:
        #   1) Detections which have not beed associated with a track: For these, create a new filter class and initialize its state based on the detection
        #   2) Tracks which have a detection: The state of these can be updated based on the associated detection
        #   3) Tracks which have no detection: It makes sense to allow for a few missing frames, nonetheless it is still necessary to predict the
        #      current filter state (e.g. based on the optical flow measurement and the object velocity). If too many frames are missing, the track can be deleted

        # Note: You can access data["detections"] and data["classes"] to receive the current list of detections and their corresponding classes
        # You must return a dictionary with the given fields:
        #       "tracks":           A Nx4 NumPy Array containing a 4-dimensional state vector for each track. Similar to the detections,
        #                           the track state containts the center point (X,Y) as well as the bounding box width and height (W, H)
        #       "trackVelocities":  A Nx2 NumPy Array with an additional velocity estimate (in pixels per frame) for each track
        #       "trackAge":         A Nx1 List with the track age (number of total frames this track exists). The track age starts at
        #                           1 on track creation and increases monotonically by 1 per frame until the track is deleted.
        #       "trackClasses":     A Nx1 List of classes associated with each track. Similar to detections, the following mapping must be used
        #                               0: Ball
        #                               1: GoalKeeper
        #                               2: Player
        #                               3: Referee
        #       "trackIds":         A Nx1 List of unique IDs for each track. IDs must not be reused and be unique during the lifetime of the program.
