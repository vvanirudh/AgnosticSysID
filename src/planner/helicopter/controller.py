import numpy as np


class LinearController:
    def __init__(self, K, nominal_state, nominal_control, time_invariant=True):
        self.K = K
        self.nominal_state = nominal_state
        self.nominal_control = nominal_control
        self.time_invariant = time_invariant

    def act(self, x, t):
        return (
            self.nominal_control + self.K.dot(np.append(x - self.nominal_state, 1))
            if self.time_invariant
            else self.nominal_control[t] + self.K[t].dot(np.append(x - self.nominal_state[t], 1))
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
            self.nominal_control + alpha * self.k + self.K.dot(np.append(x - self.nominal_state, 1))
            if self.time_invariant
            else self.nominal_control[t]
            + alpha * self.k[t]
            + self.K[t].dot(np.append(x - self.nominal_state[t], 1))
        )
