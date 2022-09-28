import numpy as np


class LinearController:
    def __init__(self, K, P, nominal_state, nominal_control, time_invariant=True):
        self.K = K
        self.P = P
        self.nominal_state = nominal_state
        self.nominal_control = nominal_control
        self.time_invariant = time_invariant

    def act(self, x, t, alpha=None):
        return (
            self.nominal_control + self.K @ (np.append(x - self.nominal_state, 1))
            if self.time_invariant
            else self.nominal_control[t] + self.K[t] @ (np.append(x - self.nominal_state[t], 1))
        )


class LinearControllerWithOffset:
    def __init__(self, k, K, nominal_state, nominal_control, time_invariant=False):
        self.k = k
        self.K = K
        self.nominal_state = nominal_state
        self.nominal_control = nominal_control
        self.time_invariant = time_invariant

    def act(self, x, t, alpha=1.0):
        return (
            (1 - alpha) * self.nominal_control
            + alpha * self.k
            + self.K @ (np.append(x - (1 - alpha) * self.nominal_state, 1))
            if self.time_invariant
            else (1 - alpha) * self.nominal_control[t]
            + alpha * self.k[t]
            + self.K[t] @ (np.append(x - (1 - alpha) * self.nominal_state[t], 1))
        )


class ManualController:
    def __init__(self, controls):
        self.controls = controls

    def act(self, x, t, alpha=None):
        return self.controls[t]
