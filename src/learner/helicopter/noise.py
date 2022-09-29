import numpy as np


def get_hover_noise():
    return np.random.randn(6)


def get_tracking_noise():
    return 0.01 * np.random.randn(6)
