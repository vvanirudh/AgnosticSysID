import numpy as np


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
