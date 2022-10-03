import numpy as np

dt = 0.05  # time scale for euler integration
g = 9.81  # gravity


class HelicopterModel:
    def __init__(self):
        ## Mass and inertia
        self.m = 5  # kg
        self.Ixx = 0.3
        self.Iyy = 0.3
        self.Izz = 0.3
        self.Ixy = self.Ixz = self.Iyz = 0

        ## Aerodynamic forces parameters
        self.Tx = np.array([0, -3.47, 13.20]) * self.Ixx
        self.Ty = np.array([0, -3.06, -9.21]) * self.Iyy
        self.Tz = np.array([0, -2.58, 14.84]) * self.Izz
        self.Fx = -0.048 * self.m
        self.Fy = np.array([0, -0.12]) * self.m
        self.Fz = np.array([-9.81, -0.0005, -27.5]) * self.m

    @property
    def I(self):
        return np.array(
            [
                [self.Ixx, self.Ixy, self.Ixz],
                [self.Ixy, self.Iyy, self.Iyz],
                [self.Ixz, self.Iyz, self.Izz],
            ]
        )

    @property
    def params(self):
        return np.concatenate(
            [
                [self.m, self.Ixx, self.Iyy, self.Izz],
                self.Tx,
                self.Ty,
                self.Tz,
                [self.Fx],
                self.Fy,
                self.Fz,
            ]
        )

    @staticmethod
    def unpack_params(params):
        assert params.shape[0] == 19
        return [
            params[0],
            params[1],
            params[2],
            params[3],
            params[4:7],
            params[7:10],
            params[10:13],
            params[13],
            params[14:16],
            params[16:19],
        ]


class ParameterizedHelicopterModel:
    def __init__(self, m, Ixx, Iyy, Izz, Tx, Ty, Tz, Fx, Fy, Fz):
        self.m = 6  # m
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.Ixy = self.Ixz = self.Iyz = 0

        self.Tx = Tx * self.Ixx
        self.Ty = Ty * self.Iyy
        self.Tz = Tz * self.Izz

        self.Fx = Fx * m
        self.Fy = Fy * m
        self.Fz = Fz * m

    @property
    def I(self):
        return np.array(
            [
                [self.Ixx, self.Ixy, self.Ixz],
                [self.Ixy, self.Iyy, self.Iyz],
                [self.Ixz, self.Iyz, self.Izz],
            ]
        )

    @property
    def params(self):
        return np.concatenate(
            [
                [self.m, self.Ixx, self.Iyy, self.Izz],
                self.Tx,
                self.Ty,
                self.Tz,
                [self.Fx],
                self.Fy,
                self.Fz,
            ]
        )

    @staticmethod
    def unpack_params(params):
        assert params.shape[0] == 19
        return [
            params[0],
            params[1],
            params[2],
            params[3],
            params[4:7],
            params[7:10],
            params[10:13],
            params[13],
            params[14:16],
            params[16:19],
        ]


class HelicopterIndex:
    def __init__(self):
        k = 0
        self.ned_dot = np.arange(k, k + 3)
        k += 3
        self.ned = np.arange(k, k + 3)
        k += 3
        self.pqr = np.arange(k, k + 3)
        k += 3
        self.axis_angle = np.arange(k, k + 3)


class LinearizedHelicopterModel:
    def __init__(self, A, B, time_varying):
        self.A = A
        self.B = B
        self.time_varying = time_varying
