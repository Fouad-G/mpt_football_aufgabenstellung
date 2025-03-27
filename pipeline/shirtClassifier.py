import numpy as np
from sklearn.cluster import KMeans


class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier"
        self.reference_colors = None  # Save cluster centers from first good frame
        self.reference_labels = None  # Save label to team mapping

    def start(self, data):
        print("[INFO] Shirt Classifier wurde gestartet.")

    def stop(self, data):
        print("[INFO] Shirt Classifier wurde gestoppt.")

    def step(self, data):
        """
        Automatische Zweiteilung der (Spieler-)Trikot-Farben mittels K-Means (n_clusters=2).
        Farblich konsistente Teamzuordnung:
        - Team A ist immer ROT
        - Team B ist immer BLAU
        """
        frame = data.get("image", None)
        tracks = data.get("tracks", np.zeros((0, 4)))
        track_classes = data.get("trackClasses", [])

        # Feste Teamfarben (BGR)
        team_a_color = (0, 0, 255)  # ROT
        team_b_color = (255, 0, 0)  # BLAU

        team_classes = [0] * len(tracks)

        if frame is None or len(tracks) == 0:
            return {
                "teamAColor": team_a_color,
                "teamBColor": team_b_color,
                "teamClasses": team_classes,
            }

        height, width, _ = frame.shape
        player_colors = []
        player_indices = []

        for i, (x_center, y_center, w, h) in enumerate(tracks):
            cls = track_classes[i] if i < len(track_classes) else 0
            if cls not in [1, 2]:
                team_classes[i] = 0
                continue

            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            if x2 <= x1 or y2 <= y1:
                team_classes[i] = 0
                continue

            tshirt_y2 = y1 + (y2 - y1) // 2
            roi = frame[y1:tshirt_y2, x1:x2]
            if roi.size == 0:
                team_classes[i] = 0
                continue

            avg_color = roi.mean(axis=(0, 1))
            player_colors.append(avg_color)
            player_indices.append(i)

        if len(player_colors) < 2:
            return {
                "teamAColor": team_a_color,
                "teamBColor": team_b_color,
                "teamClasses": team_classes,
            }

        X = np.array(player_colors, dtype=np.float32)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        desired_team_a_color = np.array([0, 0, 255], dtype=np.float32)  # ROT
        desired_team_b_color = np.array([255, 0, 0], dtype=np.float32)  # BLAU

        dist_to_a = np.linalg.norm(centers - desired_team_a_color, axis=1)
        dist_to_b = np.linalg.norm(centers - desired_team_b_color, axis=1)

        if np.argmin(dist_to_a) != np.argmin(dist_to_b):
            cluster_for_a = int(np.argmin(dist_to_a))
            cluster_for_b = int(np.argmin(dist_to_b))
            label_map = {
                cluster_for_a: 1,  # Team A (rot)
                cluster_for_b: -1,  # Team B (blau)
            }
        else:
            label_map = {0: 1, 1: -1}

        for idx, cluster_label in enumerate(labels):
            track_index = player_indices[idx]
            team_classes[track_index] = label_map[cluster_label]

        result = {
            "teamAColor": team_a_color,
            "teamBColor": team_b_color,
            "teamClasses": team_classes,
        }

        return result

    # def step(self, data):
    #     # TODO: Implement processing of a current frame list
    #     # The task of the shirt classifier module is to identify the two teams based on their shirt color and to assign each player to one of the two teams

    #     # Note: You can access data["image"] and data["tracks"] to receive the current image as well as the current track list
    #     # You must return a dictionary with the given fields:
    #     #       "teamAColor":       A 3-tuple (B, G, R) containing the blue, green and red channel values (between 0 and 255) for team A
    #     #       "teamBColor":       A 3-tuple (B, G, R) containing the blue, green and red channel values (between 0 and 255) for team B
    #     #       "teamClasses"       A list with an integer class for each track according to the following mapping:
    #     #           0: Team not decided or not a player (e.g. ball, goal keeper, referee)
    #     #           1: Player belongs to team A
    #     #           2: Player belongs to team B
    #     return { "teamAColor": None,
    #              "teamBColor": None,
    #              "teamClasses": None }
