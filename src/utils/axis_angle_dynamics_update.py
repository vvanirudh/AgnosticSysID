from src.utils.quaternion_from_axis_rotation import (
    quaternion_from_axis_rotation,
    quaternion_from_axis_rotation_batch,
)
from src.utils.quat_multiply import quat_multiply, quat_multiply_batch
from src.utils.axis_rotation_from_quaternion import (
    axis_rotation_from_quaternion,
    axis_rotation_from_quaternion_batch,
)
from numba import njit


@njit
def axis_angle_dynamics_update(axis_angle0, pqr_times_dt):
    q0 = quaternion_from_axis_rotation(axis_angle0)
    q1 = quat_multiply(q0, quaternion_from_axis_rotation(pqr_times_dt))
    return axis_rotation_from_quaternion(q1)


@njit
def axis_angle_dynamics_update_batch(axis_angle0_batch, pqr_times_dt_batch):
    q0_batch = quaternion_from_axis_rotation_batch(axis_angle0_batch)
    q1_batch = quat_multiply_batch(
        q0_batch, quaternion_from_axis_rotation_batch(pqr_times_dt_batch)
    )
    return axis_rotation_from_quaternion_batch(q1_batch)
