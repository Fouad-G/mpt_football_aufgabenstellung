import numpy as np
import cv2
from sklearn.cluster import KMeans

class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier" # Do not change the name of the module as otherwise recording replay would break!
        self.initialized = False
        self.teamAColor = None
        self.teamBColor = None

    def start(self, data):
        self.initialized = True

    def stop(self, data):
        self.initialized = False

    def step(self, data):
        image = data["image"]
        tracks = data["tracks"]
        track_classes = data.get("trackClasses", [2] * len(tracks))  # fallback if not provided

        player_indices = [i for i, cls in enumerate(track_classes) if cls == 2]
        shirt_colors = []

        for i in player_indices:
            x, y, w, h = tracks[i].astype(int)
            x1 = max(x - w // 4, 0)
            y1 = max(y - h // 4, 0)
            x2 = min(x + w // 4, image.shape[1])
            y2 = min(y + h // 4, image.shape[0])
            roi = image[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            avg_color = np.mean(roi.reshape(-1, 3), axis=0)
            if not np.isnan(avg_color).any():
                shirt_colors.append((i, avg_color))

        if len(shirt_colors) < 2:
            return {
                "teamAColor": None,
                "teamBColor": None,
                "teamClasses": [0] * len(tracks)
            }

        if self.teamAColor is None or self.teamBColor is None:
            colors = np.array([color for _, color in shirt_colors])
            kmeans = KMeans(n_clusters=2, n_init="auto", random_state=42).fit(colors)
            labels = kmeans.labels_

            if np.sum(labels == 0) >= np.sum(labels == 1):
                self.teamAColor, self.teamBColor = kmeans.cluster_centers_
            else:
                self.teamBColor, self.teamAColor = kmeans.cluster_centers_

        team_classes = [0] * len(tracks)

        for i, color in shirt_colors:
            distA = np.linalg.norm(color - self.teamAColor)
            distB = np.linalg.norm(color - self.teamBColor)

            if min(distA, distB) > 100:
                team_classes[i] = 0
            else:
                team_classes[i] = 1 if distA < distB else 2

        if team_classes.count(1) == 0 and team_classes.count(2) > 0:
            for i, c in enumerate(team_classes):
                if c == 2:
                    team_classes[i] = 1
                    break
        elif team_classes.count(2) == 0 and team_classes.count(1) > 0:
            for i, c in enumerate(team_classes):
                if c == 1:
                    team_classes[i] = 2
                    break

        team_classes_ui = [(-1 if c == 2 else c) for c in team_classes]

        return {
            "teamAColor": tuple(map(int, self.teamAColor)),
            "teamBColor": tuple(map(int, self.teamBColor)),
            "teamClasses": team_classes_ui
        }
