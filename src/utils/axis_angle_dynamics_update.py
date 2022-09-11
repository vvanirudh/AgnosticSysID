from src.utils.quaternion_from_axis_rotation import quaternion_from_axis_rotation
from src.utils.quat_multiply import quat_multiply
from src.utils.axis_rotation_from_quaternion import axis_rotation_from_quaternion


def axis_angle_dynamics_update(axis_angle0, pqr_times_dt):
    q0 = quaternion_from_axis_rotation(axis_angle0)
    q1 = quat_multiply(q0, quaternion_from_axis_rotation(pqr_times_dt))
    return axis_rotation_from_quaternion(q1)
