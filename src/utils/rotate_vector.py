from src.utils.quat_multiply import quat_multiply, quat_multiply_batch
import numpy as np


def rotate_vector(vin, q):
    return quat_multiply(
        quat_multiply(q, np.array([vin[0], vin[1], vin[2], 0])),
        np.array([-q[0], -q[1], -q[2], q[3]]),
    )[0:3]


def rotate_vector_batch(vin_batch, q_batch):
    q_transformed_batch = -1 * q_batch.copy()
    q_transformed_batch[:, 3] *= -1
    vin_transformed_batch = np.hstack((vin_batch.copy(), np.zeros((vin_batch.shape[0], 1))))
    return quat_multiply_batch(
        quat_multiply_batch(q_batch, vin_transformed_batch), q_transformed_batch
    )[:, 0:3]
