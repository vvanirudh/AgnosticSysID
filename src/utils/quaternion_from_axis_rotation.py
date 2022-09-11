import numpy as np


def quaternion_from_axis_rotation(axis_rotation):
    rotation_angle = np.linalg.norm(axis_rotation)
    quat = np.zeros(4)

    if rotation_angle < 1e-4:
        quat[0:3] = axis_rotation / 2
    else:
        normalized_axis = axis_rotation / rotation_angle
        quat[0:3] = normalized_axis * np.sin(rotation_angle / 2)

    quat[3] = np.sqrt(1 - np.linalg.norm(quat[0:3]) ** 2)

    return quat
