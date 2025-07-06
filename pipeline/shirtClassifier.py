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
        # TODO: Implement start up procedure of the module
        print("[ShirtClassifier] Module started.")
        self.initialized = True

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        print("[ShirtClassifier] Module stopped.")
        self.initialized = False

    def step(self, data):
        # TODO: Implement processing of a current frame list
        # The task of the shirt classifier module is to identify the two teams based on their shirt color and to assign each player to one of the two teams

        # Note: You can access data["image"] and data["tracks"] to receive the current image as well as the current track list
        # You must return a dictionary with the given fields:
        #       "teamAColor":       A 3-tuple (B, G, R) containing the blue, green and red channel values (between 0 and 255) for team A
        #       "teamBColor":       A 3-tuple (B, G, R) containing the blue, green and red channel values (between 0 and 255) for team B
        #       "teamClasses"       A list with an integer class for each track according to the following mapping:
        #           0: Team not decided or not a player (e.g. ball, goal keeper, referee)
        #           1: Player belongs to team A
        #           2: Player belongs to team B

        image = data["image"]
        tracks = data["tracks"]
        track_classes = data.get("trackClasses", [2] * len(tracks))  # fallback falls nicht gesetzt

        player_indices = []
        shirt_colors = []

        for i, cls in enumerate(track_classes):
            if cls == 2:
                player_indices.append(i)

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
                "teamClasses": [0 for _ in data["tracks"]]
            }

        # Recompute cluster colors on every frame using KMeans
        colors = np.array([color for _, color in shirt_colors])
        kmeans = KMeans(n_clusters=2, n_init="auto", random_state=42).fit(colors)
        labels = kmeans.labels_

        # Assign the more frequent cluster as team A
        count0 = np.sum(labels == 0)
        count1 = np.sum(labels == 1)

        if count0 >= count1:
            self.teamAColor = kmeans.cluster_centers_[0]
            self.teamBColor = kmeans.cluster_centers_[1]
        else:
            self.teamAColor = kmeans.cluster_centers_[1]
            self.teamBColor = kmeans.cluster_centers_[0]

        print("Team A Color:", self.teamAColor)
        print("Team B Color:", self.teamBColor)

        # Assign each player to the closest team color
        team_classes = [0] * len(tracks)
        for i, color in shirt_colors:
            distA = np.linalg.norm(color - self.teamAColor)
            distB = np.linalg.norm(color - self.teamBColor)

            print(f"Player {i}: Color {color}, Distance to A: {distA:.2f}, Distance to B: {distB:.2f}")

            if min(distA, distB) > 100:  # Distance threshold: unclear team assignment
                team_classes[i] = 0
            else:
                team_classes[i] = 1 if distA < distB else 2

        # print("[ShirtClassifier] Spieler gefunden:", player_indices)
        # print("Frame received")
        # print("Number of tracks:", len(tracks))
        return {
            "teamAColor": tuple(map(int, self.teamAColor)),
            "teamBColor": tuple(map(int, self.teamBColor)),
            "teamClasses": team_classes
        }
