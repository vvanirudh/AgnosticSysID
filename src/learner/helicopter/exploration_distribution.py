import numpy as np
from src.planner.helicopter.helicopter_hover import hover_at_zero, hover_trims, hover_controller
from src.env.helicopter.helicopter_model import dt


class ExplorationDistribution:
    def __init__(self, horizon, type="Gaussian"):
        self.horizon = horizon
        self.type = type
        self.mean_states = [None for _ in range(self.horizon)]
        self.mean_controls = [None for _ in range(self.horizon)]
        self.noise_states = [None for _ in range(self.horizon)]
        self.noise_controls = [None for _ in range(self.horizon)]

    def update(self, mean_state, mean_control, noise_state, noise_control, t):
        assert 0 <= t < self.horizon
        self.mean_states[t] = mean_state
        self.mean_controls[t] = mean_control
        self.noise_states[t] = noise_state
        self.noise_controls[t] = noise_control

    def sample(self, t):
        assert 0 <= t < self.horizon
        return np.random.normal(self.mean_states[t], self.noise_states[t]), np.random.normal(
            self.mean_controls[t], self.noise_controls[t]
        )


def desired_trajectory_exploration_distribution(H, noise_state, noise_control):
    exploration_distribution = ExplorationDistribution(H)
    for t in range(H):
        exploration_distribution.update(hover_at_zero, hover_trims, noise_state, noise_control, t)
    return exploration_distribution


def expert_exploration_distribution(
    helicopter_env, helicopter_model, helicopter_index, H, noise_state, noise_control
):
    expert_controller = hover_controller(helicopter_model, helicopter_index, helicopter_env)
    exploration_distribution = ExplorationDistribution(H)
    state = hover_at_zero.copy()
    for t in range(H):
        control = expert_controller.act(state, t)
        exploration_distribution.update(state, control, noise_state, noise_control, t)
        # TODO: Should I be adding noise when rolling out expert controller?
        state = helicopter_env.step(
            state, control, dt, helicopter_model, helicopter_index, noise=np.random.randn(6)
        )

    return exploration_distribution
