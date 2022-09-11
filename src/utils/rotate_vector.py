from src.utils.quat_multiply import quat_multiply
import numpy as np


def rotate_vector(vin, q):
    return quat_multiply(
        quat_multiply(q, np.array([vin[0], vin[1], vin[2], 0])),
        np.array([-q[0], -q[1], -q[2], q[3]]),
    )[0:3]
