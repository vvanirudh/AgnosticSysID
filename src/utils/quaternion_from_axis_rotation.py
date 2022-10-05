from cgitb import small
import numpy as np
from numba import njit


@njit
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


@njit
def quaternion_from_axis_rotation_batch(axis_rotation_batch):
    # rotation_angle_batch = np.linalg.norm(axis_rotation_batch, axis=1)
    rotation_angle_batch = np.array(
        [np.linalg.norm(axis_rotation_batch[i, :]) for i in range(axis_rotation_batch.shape[0])]
    )
    quat_batch = np.zeros((axis_rotation_batch.shape[0], 4))

    small_angle_idxs = rotation_angle_batch < 1e-4
    quat_batch[small_angle_idxs, 0:3] = axis_rotation_batch[small_angle_idxs, :] / 2

    large_angle_idxs = np.logical_not(small_angle_idxs)
    normalized_axis_batch = axis_rotation_batch[large_angle_idxs, :] / rotation_angle_batch[
        large_angle_idxs
    ].reshape(-1, 1)
    quat_batch[large_angle_idxs, 0:3] = normalized_axis_batch * np.sin(
        rotation_angle_batch[large_angle_idxs]
    ).reshape(-1, 1)

    quat_batch_norm = np.array(
        [np.linalg.norm(quat_batch[i, 0:3]) for i in range(quat_batch.shape[0])]
    )
    quat_batch[:, 3] = np.sqrt(1 - quat_batch_norm**2)
    # quat_batch[:, 3] = np.sqrt(1 - np.linalg.norm(quat_batch[:, 0:3], axis=1) ** 2)

    return quat_batch
