import numpy as np
from numba import njit


@njit
def axis_rotation_from_quaternion(q):
    q_norm = np.linalg.norm(q[0:3])
    rotation_angle = np.arcsin(q_norm) * 2

    eps = 1e-6
    if rotation_angle < eps:
        return np.zeros(3)
    return q[0:3] / q_norm * rotation_angle


@njit
def axis_rotation_from_quaternion_batch(q_batch):
    q_norm_batch = np.array([np.linalg.norm(q_batch[i, 0:3]) for i in range(q_batch.shape[0])])
    # q_norm_batch = np.linalg.norm(q_batch[:, 0:3], axis=1)
    rotation_angle_batch = np.arcsin(q_norm_batch) * 2

    eps = 1e-6
    small_angle_idxs = rotation_angle_batch < eps
    large_angle_idxs = np.logical_not(small_angle_idxs)

    axis_rotation_batch = q_batch[:, 0:3].copy()
    axis_rotation_batch[small_angle_idxs, :] = 0

    axis_rotation_batch[large_angle_idxs, :] = axis_rotation_batch[
        large_angle_idxs, :
    ] / q_norm_batch[large_angle_idxs].reshape(-1, 1)
    axis_rotation_batch[large_angle_idxs, :] = axis_rotation_batch[
        large_angle_idxs, :
    ] * rotation_angle_batch[large_angle_idxs].reshape(-1, 1)

    return axis_rotation_batch
