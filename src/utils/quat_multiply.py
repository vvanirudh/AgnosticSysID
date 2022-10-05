import numpy as np
from numba import njit


@njit
def quat_multiply(lq, rq):
    quat = np.zeros(4)

    quat[0] = lq[3] * rq[0] + lq[0] * rq[3] + lq[1] * rq[2] - lq[2] * rq[1]
    quat[1] = lq[3] * rq[1] - lq[0] * rq[2] + lq[1] * rq[3] + lq[2] * rq[0]
    quat[2] = lq[3] * rq[2] + lq[0] * rq[1] - lq[1] * rq[0] + lq[2] * rq[3]
    quat[3] = lq[3] * rq[3] - lq[0] * rq[0] - lq[1] * rq[1] - lq[2] * rq[2]

    """ quat(1) = lq(4)*rq(1) + lq(1)*rq(4) + lq(2)*rq(3) - lq(3)*rq(2);
    quat(2) = lq(4)*rq(2) - lq(1)*rq(3) + lq(2)*rq(4) + lq(3)*rq(1);
    quat(3) = lq(4)*rq(3) + lq(1)*rq(2) - lq(2)*rq(1) + lq(3)*rq(4);
    quat(4) = lq(4)*rq(4) - lq(1)*rq(1) - lq(2)*rq(2) - lq(3)*rq(3); """

    return quat


@njit
def quat_multiply_batch(lq_batch, rq_batch):
    quat_batch = np.zeros((lq_batch.shape[0], 4))

    quat_batch[:, 0] = (
        lq_batch[:, 3] * rq_batch[:, 0]
        + lq_batch[:, 0] * rq_batch[:, 3]
        + lq_batch[:, 1] * rq_batch[:, 2]
        - lq_batch[:, 2] * rq_batch[:, 1]
    )
    quat_batch[:, 1] = (
        lq_batch[:, 3] * rq_batch[:, 1]
        - lq_batch[:, 0] * rq_batch[:, 2]
        + lq_batch[:, 1] * rq_batch[:, 3]
        + lq_batch[:, 2] * rq_batch[:, 0]
    )
    quat_batch[:, 2] = (
        lq_batch[:, 3] * rq_batch[:, 2]
        + lq_batch[:, 0] * rq_batch[:, 1]
        - lq_batch[:, 1] * rq_batch[:, 0]
        + lq_batch[:, 2] * rq_batch[:, 3]
    )
    quat_batch[:, 3] = (
        lq_batch[:, 3] * rq_batch[:, 3]
        - lq_batch[:, 0] * rq_batch[:, 0]
        - lq_batch[:, 1] * rq_batch[:, 1]
        - lq_batch[:, 2] * rq_batch[:, 2]
    )

    return quat_batch
