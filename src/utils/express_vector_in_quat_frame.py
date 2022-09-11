from src.utils.rotate_vector import rotate_vector
import numpy as np


def express_vector_in_quat_frame(vin, q):
    return rotate_vector(vin, np.array([-q[0], -q[1], -q[2], q[3]]))
