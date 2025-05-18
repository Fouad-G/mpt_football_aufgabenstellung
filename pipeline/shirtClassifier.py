class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier" # Do not change the name of the module as otherwise recording replay would break!

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
        track_classes = data.get("trackClasses", [])

        player_indices = []
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


        # print("[ShirtClassifier] Spieler gefunden:", player_indices)
        # print("Frame received")
        # print("Number of tracks:", len(tracks))
        return { "teamAColor": (0, 0, 255),
                 "teamBColor": (0, 255, 0),
                 "teamClasses": [0 for _ in data["tracks"]] }