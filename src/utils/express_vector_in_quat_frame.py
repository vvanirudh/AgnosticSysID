from src.utils.rotate_vector import rotate_vector, rotate_vector_batch
import numpy as np
from numba import njit


@njit
def express_vector_in_quat_frame(vin, q):
    return rotate_vector(vin, np.array([-q[0], -q[1], -q[2], q[3]]))


@njit
def express_vector_in_quat_frame_batch(vin_batch, q_batch):
    q_transformed_batch = -1 * q_batch.copy()
    q_transformed_batch[:, 3] *= -1
    return rotate_vector_batch(vin_batch, q_transformed_batch)
