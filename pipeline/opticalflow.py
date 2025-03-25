import cv2 as cv
import numpy as np

class OpticalFlow:
    def __init__(self, scale_factor=0.5, mirror=True, use_gpu=False):
        """
        :param scale_factor: Skalierungsfaktor für schnellere Verarbeitung (z. B. 0.5 für 50% Größe)
        :param mirror: True = Bild wird horizontal gespiegelt
        :param use_gpu: True = CUDA-basierter Optical Flow (nur mit NVIDIA GPU)
        """
        self.name = "Optical Flow"
        self.prev_gray = None
        self.scale_factor = scale_factor
        self.mirror = mirror
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.gpu_flow = cv.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)

    def start(self, data):
        self.prev_gray = None

    def stop(self, data):
        self.prev_gray = None

    def step(self, data):
        frame = data.get("image", None)
        if frame is None:
            return {"opticalFlow": None}

        if self.scale_factor != 1.0:
            frame = cv.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv.INTER_LINEAR)

        if self.mirror:
            frame = cv.flip(frame, 1)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return {"opticalFlow": np.array([0, 0])}

        if self.use_gpu:
            d_prev_gray = cv.cuda_GpuMat(self.prev_gray)
            d_gray = cv.cuda_GpuMat(gray)
            d_flow = self.gpu_flow.calc(d_prev_gray, d_gray, None)
            flow = d_flow.download()
        else:
            flow = cv.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        avg_flow = np.mean(flow, axis=(0, 1)) / self.scale_factor

        self.prev_gray = gray
        return {"opticalFlow": avg_flow}
