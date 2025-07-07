import numpy as np
import cv2
from sklearn.cluster import KMeans

class ShirtClassifier:
    """
    A module that classifies players into two teams based on shirt color.

    It detects two dominant colors from the player bounding boxes and assigns
    each track (object) to one of three categories:
      - 0: Not a player or undecided
      - 1: Team A
      - -1: Team B (for UI compatibility)
    """

    def __init__(self):
        """
        Initializes the ShirtClassifier.
        """
        self.name = "Shirt Classifier"  # Do not change this name, required for replay compatibility
        self.initialized = False
        self.teamAColor = None  # Mean color of Team A
        self.teamBColor = None  # Mean color of Team B

    def start(self, data):
        """
        Called once at module start-up.
        """
        self.initialized = True

    def stop(self, data):
        """
        Called once at module shutdown.
        """
        self.initialized = False

    def step(self, data):
        """
        Processes one frame of input data to classify players by shirt color.

        Args:
            data (dict): Must contain the following keys:
                - "image": The current RGB image as a NumPy array (H x W x 3)
                - "tracks": A list of bounding boxes (each as np.array([x, y, w, h]))
                - "trackClasses": (optional) A list of object classes; class 2 means "player"

        Returns:
            dict: A dictionary with:
                - "teamAColor": Tuple of BGR values for Team A (e.g., (100, 0, 200))
                - "teamBColor": Tuple of BGR values for Team B
                - "teamClasses": List of integers with:
                    0 = undecided or not a player,
                    1 = Team A,
                   -1 = Team B (as expected by the UI)
        """
        image = data["image"]
        tracks = data["tracks"]
        track_classes = data.get("trackClasses", [2] * len(tracks))  # Default to all players if missing

        # Identify indices of all player-class tracks (class == 2)
        player_indices = [i for i, cls in enumerate(track_classes) if cls == 2]
        shirt_colors = []

        # Extract average shirt color from ROI around each player's upper body
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

        # Not enough valid colors detected
        if len(shirt_colors) < 2:
            return {
                "teamAColor": None,
                "teamBColor": None,
                "teamClasses": [0] * len(tracks)
            }

        # If team colors are not yet set, perform clustering using KMeans
        if self.teamAColor is None or self.teamBColor is None:
            colors = np.array([color for _, color in shirt_colors])
            kmeans = KMeans(n_clusters=2, n_init="auto", random_state=42).fit(colors)
            labels = kmeans.labels_

            # Assign the more frequent cluster to Team A
            if np.sum(labels == 0) >= np.sum(labels == 1):
                self.teamAColor, self.teamBColor = kmeans.cluster_centers_
            else:
                self.teamBColor, self.teamAColor = kmeans.cluster_centers_

        # Initialize all tracks as undecided
        team_classes = [0] * len(tracks)

        # Assign players to Team A or Team B based on color distance
        for i, color in shirt_colors:
            distA = np.linalg.norm(color - self.teamAColor)
            distB = np.linalg.norm(color - self.teamBColor)

            if min(distA, distB) > 100:  # Too far from either color --> undecided
                team_classes[i] = 0
            else:
                team_classes[i] = 1 if distA < distB else 2

        # UI visibility fix: ensure at least one player per team is assigned
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

        # Convert Team B (2) to -1 for UI compatibility
        team_classes_ui = [(-1 if c == 2 else c) for c in team_classes]

        return {
            "teamAColor": tuple(map(int, self.teamAColor)),
            "teamBColor": tuple(map(int, self.teamBColor)),
            "teamClasses": team_classes_ui
        }
