import cv2 as cv
import numpy as np

class OpticalFlow:
    """
    Optischer Fluss , berechnet die durchschnittliche Pixelverschiebung
    zwischen aufeinanderfolgenden Bildern.
    """

    def __init__(self, scale_factor=0.5, mirror=True, use_gpu=False):
        """
        :param scale_factor: Faktor zur Verkleinerung der Bilder für mehr Geschwindigkeit.
        :param mirror: Ob das Bild horizontal gespiegelt werden soll.
        :param use_gpu: Ob die CUDA-Version verwendet werden soll, falls verfügbar.
        """
        self.name = "Optical Flow"
        self.prev_gray = None
        self.scale_factor = scale_factor
        self.mirror = mirror
        self.use_gpu = use_gpu  

        if self.use_gpu:
            try:
                self.gpu_flow = cv.cuda_FarnebackOpticalFlow.create(
                    numLevels=5,
                    pyrScale=0.5,
                    fastPyramids=False,
                    winSize=15,
                    numIters=3,
                    polyN=5,
                    polySigma=1.2,
                    flags=0,
                )
            except AttributeError:
                self.use_gpu = False
        
    def start(self, data):
        self.prev_gray = None

    def stop(self, data):
        self.prev_gray = None
    
    def step(self, data):
        """
        Verarbeitet ein Bild und gibt den durchschnittlichen Bewegungsvektor zurück.
        :param data: Dictionary mit dem Schlüssel 'image' für das BGR-Bild.
        :return: {'opticalFlow': 1x2 np.ndarray oder None}
        """
        frame = data.get("image")
        if frame is None:
            return {"opticalFlow": None}

        if self.scale_factor != 1.0:
            frame = cv.resize(frame, (0, 0),
                              fx=self.scale_factor,
                              fy=self.scale_factor,
                              interpolation=cv.INTER_LINEAR)

        if self.mirror:
            frame = cv.flip(frame, 1)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


