import cv2 as cv
import numpy as np


class OpticalFlow:
    """
    Optical Flow module that calculates the average pixel displacement
    between consecutive frames.
    """

    def __init__(self, scale_factor=0.5, mirror=True, use_gpu=False):
        """
        Initializes the OpticalFlow module.

        Parameters:
            scale_factor (float): Downscale factor to speed up processing.
            mirror (bool): Whether to horizontally flip the image.
            use_gpu (bool): Whether to use the CUDA implementation if available.
        """
        self.name = "Optical Flow"
        self.prev_gray = None
        self.scale_factor = scale_factor
        self.mirror = mirror
        self.use_gpu = use_gpu

        # Try to initialize GPU-based optical flow if enabled
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
                # Fallback to CPU if GPU is not available
                self.use_gpu = False

    def start(self, data):
        """
        Called once at module startup.
        """
        self.prev_gray = None

    def stop(self, data):
        """
        Called once at module shutdown.
        """
        self.prev_gray = None

    def step(self, data):
        """
        Processes an image and returns the average motion vector.

        Parameters:
            data (dict): Dictionary containing the key 'image' (BGR image).

        Returns:
            dict: {'opticalFlow': np.ndarray of shape (2,) or None}
        """
        frame = data.get("image")
        if frame is None:
            return {"opticalFlow": None}

        # Resize image if scaling is enabled
        if self.scale_factor != 1.0:
            frame = cv.resize(
                frame,
                (0, 0),
                fx=self.scale_factor,
                fy=self.scale_factor,
                interpolation=cv.INTER_LINEAR,
            )

        # Flip image horizontally if enabled
        if self.mirror:
            frame = cv.flip(frame, 1)

        # Convert frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Initialize previous frame if not set
        if self.prev_gray is None:
            self.prev_gray = gray
            return {"opticalFlow": np.zeros((2,), dtype=np.float32)}

        # Compute optical flow
        if self.use_gpu:
            try:
                d_prev = cv.cuda_GpuMat()
                d_prev.upload(self.prev_gray)
                d_curr = cv.cuda_GpuMat()
                d_curr.upload(gray)
                d_flow = self.gpu_flow.calc(d_prev, d_curr, None)
                flow = d_flow.download()
            except Exception:
                # Fallback to CPU if GPU fails
                self.use_gpu = False
                flow = cv.calcOpticalFlowFarneback(
                    self.prev_gray, gray, None, 0.5, 5, 15, 3, 5, 1.2, 0
                )
        else:
            flow = cv.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 5, 15, 3, 5, 1.2, 0
            )

        # Compute average flow vector (dx, dy), rescale to original size
        avg_flow = np.mean(flow, axis=(0, 1)) / self.scale_factor

        # Update previous frame
        self.prev_gray = gray

        return {"opticalFlow": avg_flow}
