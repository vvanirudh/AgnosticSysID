from src.env.helicopter.helicopter_model import (
    HelicopterIndex,
    HelicopterModel,
    ParameterizedHelicopterModel,
)
from src.utils.quaternion_from_axis_rotation import (
    quaternion_from_axis_rotation,
    quaternion_from_axis_rotation_batch,
)
from src.utils.axis_angle_dynamics_update import (
    axis_angle_dynamics_update,
    axis_angle_dynamics_update_batch,
)
from src.utils.rotate_vector import rotate_vector, rotate_vector_batch
from src.utils.express_vector_in_quat_frame import (
    express_vector_in_quat_frame,
    express_vector_in_quat_frame_batch,
)
import numpy as np
import ray

CONTROL_LIMITS = 1e5


class HelicopterEnv:
    def __init__(self):
        pass

    def step(self, x0, u0, dt, helicopter_model, helicopter_index, noise=None):
        uclipped = np.clip(u0, -CONTROL_LIMITS, CONTROL_LIMITS)
        Fned, Txyz = self.compute_forces_and_torques(
            x0, uclipped, helicopter_model, helicopter_index
        )

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
        quat = quaternion_from_axis_rotation(x0[helicopter_index.axis_angle])
        uvw = express_vector_in_quat_frame(
            x0[helicopter_index.ned_dot],
            quat,
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
        F_ned_minus_g = rotate_vector(Fxyz_minus_g, quat)

        # add gravity to complete the forces
        # if isinstance(helicopter_model, HelicopterModel):
        #     Fned = F_ned_minus_g + helicopter_model.m * np.array([0, 0, 9.81])
        # else:
        #     Fned = F_ned_minus_g.copy()
        Fned = F_ned_minus_g + helicopter_model.m * np.array([0, 0, helicopter_model.g])

        ## Torques
        Txyz = np.array(
            [
                helicopter_model.Tx @ np.array([1, x0[helicopter_index.pqr[0]], u0[0]]),
                helicopter_model.Ty @ np.array([1, x0[helicopter_index.pqr[1]], u0[1]]),
                helicopter_model.Tz @ np.array([1, x0[helicopter_index.pqr[2]], u0[2]]),
            ]
        )

        return Fned, Txyz

    def step_batch(self, X, U, dt, helicopter_model, helicopter_index, noise=None):
        Uclipped = np.clip(U, -CONTROL_LIMITS, CONTROL_LIMITS)
        FNED, TXYZ = self.compute_forces_and_torques_batch(
            X, Uclipped, helicopter_model, helicopter_index
        )

        if noise is not None:
            FNED = FNED + noise[0:3]
            TXYZ = TXYZ + noise[3:6]

        X1 = X.copy()
        X1[:, helicopter_index.ned_dot] += dt * FNED / helicopter_model.m
        X1[:, helicopter_index.pqr] += (
            dt * TXYZ / np.array([helicopter_model.Ixx, helicopter_model.Iyy, helicopter_model.Izz])
        )

        X1[:, helicopter_index.ned] += dt * X[:, helicopter_index.ned_dot]
        X1[:, helicopter_index.axis_angle] += axis_angle_dynamics_update_batch(
            X[:, helicopter_index.axis_angle], X[:, helicopter_index.pqr] * dt
        )

        return X1

    def compute_forces_and_torques_batch(self, X, U, helicopter_model, helicopter_index):
        QUAT = quaternion_from_axis_rotation_batch(X[:, helicopter_index.axis_angle])
        UVW = express_vector_in_quat_frame_batch(
            X[:, helicopter_index.ned_dot],
            QUAT,
        )

        UVW0 = UVW[:, 0]
        UVW1 = np.array([np.ones(UVW.shape[0]), UVW[:, 1]]).T
        UVW2 = np.array([np.ones(UVW.shape[0]), UVW[:, 2], U[:, 3]]).T

        FXYZ_MINUS_G = np.array(
            [
                UVW0 * helicopter_model.Fx,
                UVW1 @ helicopter_model.Fy,
                UVW2 @ helicopter_model.Fz,
            ]
        ).T

        F_NED_MINUS_G = rotate_vector_batch(FXYZ_MINUS_G, QUAT)

        FNED = F_NED_MINUS_G + helicopter_model.m * np.array([0, 0, 9.81])

        PQR1 = np.array([np.ones(U.shape[0]), X[:, helicopter_index.pqr[0]], U[:, 0]]).T
        PQR2 = np.array([np.ones(U.shape[0]), X[:, helicopter_index.pqr[1]], U[:, 1]]).T
        PQR3 = np.array([np.ones(U.shape[0]), X[:, helicopter_index.pqr[2]], U[:, 2]]).T

        TXYZ = np.array(
            [
                PQR1 @ helicopter_model.Tx,
                PQR2 @ helicopter_model.Ty,
                PQR3 @ helicopter_model.Tz,
            ]
        ).T

        return FNED, TXYZ


class LinearizedHelicopterEnv:
    def __init__(self, time_varying):
        self.time_varying = time_varying

    def step(self, x0, u0, linearized_helicopter_model, t=None):
        if self.time_varying and t is None:
            raise Exception("Time varying env needs t to be specified")

        if self.time_varying:
            return (
                linearized_helicopter_model.A[t] @ (np.append(x0, 1))
                + linearized_helicopter_model.B[t] @ u0
            )

        return (
            linearized_helicopter_model.A @ (np.append(x0, 1)) + linearized_helicopter_model.B @ u0
        )


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
