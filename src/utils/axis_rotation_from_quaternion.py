import numpy as np


def axis_rotation_from_quaternion(q):
    rotation_angle = np.arcsin(np.linalg.norm(q[0:3])) * 2

    eps = 1e-6
    if rotation_angle < eps:
        return np.zeros(3)
    return q[0:3] / np.linalg.norm(q[0:3]) * rotation_angle
