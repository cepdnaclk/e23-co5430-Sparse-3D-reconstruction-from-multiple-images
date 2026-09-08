"""Core data models for the classical SfM pipeline."""

import numpy as np


class Camera:
    def __init__(self, K):
        self.K = K.astype(np.float64)


class Frame:
    def __init__(self, idx, path, kp, desc, image_shape):
        self.idx = idx
        self.path = path
        self.kp = kp
        self.desc = desc
        self.shape = image_shape
        self.registered = False
        self.R = None
        self.t = None
        self.point3d_idx = -np.ones(len(kp), dtype=np.int64)
