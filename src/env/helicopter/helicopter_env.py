from src.env.helicopter.helicopter_model import (
    HelicopterIndex,
    HelicopterModel,
    ParameterizedHelicopterModel,
)
from src.utils.quaternion_from_axis_rotation import quaternion_from_axis_rotation
from src.utils.axis_angle_dynamics_update import axis_angle_dynamics_update
from src.utils.rotate_vector import rotate_vector
from src.utils.express_vector_in_quat_frame import express_vector_in_quat_frame
import numpy as np
import ray


class HelicopterEnv:
    def __init__(self):
        pass

    def step(self, x0, u0, dt, helicopter_model, helicopter_index, noise=None):
        Fned, Txyz = self.compute_forces_and_torques(x0, u0, helicopter_model, helicopter_index)

        ## add noise
        if noise is not None:
            Fned = Fned + noise[0:3]
            Txyz = Txyz + noise[3:6]

        ## angular rate and velocity simulation: [this ignores inertial coupling; apparently works just fine on our helicopters]
        x1 = x0.copy()
        x1[helicopter_index.ned_dot] += dt * Fned / helicopter_model.m
        x1[helicopter_index.pqr] += (
            dt * Txyz / np.array([helicopter_model.Ixx, helicopter_model.Iyy, helicopter_model.Izz])
        )

        ## position and orientation merely require integration (we use euler integration)
        x1[helicopter_index.ned] += dt * x0[helicopter_index.ned_dot]
        x1[helicopter_index.axis_angle] += axis_angle_dynamics_update(
            x0[helicopter_index.axis_angle], x0[helicopter_index.pqr] * dt
        )

        return x1

    def compute_forces_and_torques(self, x0, u0, helicopter_model, helicopter_index):
        uvw = express_vector_in_quat_frame(
            x0[helicopter_index.ned_dot],
            quaternion_from_axis_rotation(x0[helicopter_index.axis_angle]),
        )

        ## aerodynamic forces
        # expressed in heli frame:
        Fxyz_minus_g = np.array(
            [
                helicopter_model.Fx * uvw[0],
                helicopter_model.Fy @ np.array([1, uvw[1]]),
                helicopter_model.Fz @ np.array([1, uvw[2], u0[3]]),
            ]
        )

        # expressed in ned frame
        F_ned_minus_g = rotate_vector(
            Fxyz_minus_g, quaternion_from_axis_rotation(x0[helicopter_index.axis_angle])
        )

        # add gravity to complete the forces
        Fned = F_ned_minus_g + helicopter_model.m * np.array([0, 0, 9.81])

        ## Torques
        Txyz = np.array(
            [
                helicopter_model.Tx @ np.array([1, x0[helicopter_index.pqr[0]], u0[0]]),
                helicopter_model.Ty @ np.array([1, x0[helicopter_index.pqr[1]], u0[1]]),
                helicopter_model.Tz @ np.array([1, x0[helicopter_index.pqr[2]], u0[2]]),
            ]
        )

        return Fned, Txyz


class LinearizedHelicopterEnv:
    def __init__(self, time_varying):
        self.time_varying = time_varying

    def step(self, x0, u0, linearized_helicopter_model, t=None):
        if self.time_varying and t is None:
            raise Exception("Time varying env needs t to be specified")

        if self.time_varying:
            return linearized_helicopter_model.A[t] @ (
                np.append(x0, 1)
            ) + linearized_helicopter_model.B[t] @ u0

        return linearized_helicopter_model.A @ (
            np.append(x0, 1)
        ) + linearized_helicopter_model.B @ u0


def setup_env():
    helicopter_model = HelicopterModel()
    helicopter_index = HelicopterIndex()
    helicopter_env = HelicopterEnv()

    return helicopter_model, helicopter_index, helicopter_env


# Simple wrapper function
@ray.remote
def step_parameterized_model(x0, u0, dt, params):
    env = HelicopterEnv()
    model = ParameterizedHelicopterModel(*ParameterizedHelicopterModel.unpack_params(params))
    index = HelicopterIndex()

    return env.step(x0, u0, dt, model, index)


def step_batch_parameterized_model(X, U, dt, params):
    N = X.shape[0]
    return ray.get(
        [step_parameterized_model.remote(X[i, :], U[i, :], dt, params) for i in range(N)]
    )
