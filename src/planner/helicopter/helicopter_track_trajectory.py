from src.env.helicopter.helicopter_env import setup_env
from src.env.helicopter.helicopter_model import dt, HelicopterIndex
from src.env.helicopter.linearized_helicopter_dynamics import linearized_heli_dynamics_2
from src.learner.helicopter.noise import get_tracking_noise
from src.planner.helicopter.cost import cost_control, cost_final, cost_state
from src.planner.helicopter.helicopter_hover import Q, R, Qfinal
from src.planner.lqr import lqr_ltv
from src.planner.helicopter.controller import LinearController

import numpy as np
import matplotlib.pyplot as plt

hover_at_zero = np.zeros(12)
hover_trims = np.zeros(4)


def nose_in_funnel_trajectory(helicopter_index: HelicopterIndex):
    """
    This maneuver consists in rotating at
    a fixed speed and distance around an axis normal to the
    ground with the helicopter’s nose pointing towards the axis
    of rotation. We attempt to
    learn to perform 4 complete rotations of radius 5.
    """
    H = 301
    num_rotations = 4
    angular_velocity = num_rotations * 2 * np.pi / (H * dt)
    target_traj = []
    state = np.zeros(12)
    angle = -np.pi
    state[helicopter_index.ned] = np.array([5 * np.cos(angle), 5 * np.sin(angle), 5])
    state[helicopter_index.axis_angle] = np.array([0, 0, 1]) * angle

    for t in range(H):
        target_traj.append(state.copy())
        angle += angular_velocity * dt
        angle = angle if angle < np.pi else angle - 2 * np.pi
        state[helicopter_index.ned] = np.array([5 * np.cos(angle), 5 * np.sin(angle), 5])
        state[helicopter_index.axis_angle] = np.array([0, 0, 1]) * angle

    target_traj = np.array(target_traj)
    return target_traj


def desired_trajectory(helicopter_index: HelicopterIndex):
    """
    %--note the target is not exactly flyable, b/c the helicopter needs
    %to pitch and roll to follow the target positions; our approach should be
    %able to handle this though and fly as closely as possible (where close is
    %defined by our cost function)
    """
    H = 300
    hover_at_destination = np.zeros(12)
    hover_at_destination[helicopter_index.ned] = np.array([5, 0, 5])
    target_traj = []
    # First 5 seconds we hover in place
    for t in range(100):
        target_traj.append(hover_at_zero)

    # Next 5 seconds fly to destination (translation + rotation)
    for t in range(101, 201):
        desired_state = np.zeros(12)
        desired_state[helicopter_index.ned_dot] = hover_at_destination[helicopter_index.ned] / 5
        desired_state[helicopter_index.ned] = (
            hover_at_destination[helicopter_index.ned] * (t - 100.0) / 100
        )
        desired_state[helicopter_index.axis_angle] = (
            (t - 100.0) / 100 * hover_at_destination[helicopter_index.axis_angle]
        )

        target_traj.append(desired_state)

    # Last 5 seconds hover at destination
    for t in range(201, 301):
        target_traj.append(hover_at_destination)

    target_traj = np.array(target_traj)
    return target_traj


def tracking_controller(trajectory, helicopter_model, helicopter_index, helicopter_env):
    """
    %% My suggestion:
    %% (1) find A_t, B_t such that:  [x_{t+1} - x^*_{t+1} ; 1] = A_t [(x_t-x^*_t); 1] + B u_t
    %% (2) use the provided Q and R matrices and work backwards in time to find
    %% the sequence of linear feedback controllers optimal for the linearized
    %% system
    """
    H = trajectory.shape[0] - 1
    # Q = np.diag(np.append(np.ones(12), 0))
    # R = np.eye(4)
    # Qfinal = Q.copy()

    A = []
    B = []
    for t in range(H):
        A_t, B_t = linearized_heli_dynamics_2(
            trajectory[t],
            trajectory[t + 1],
            hover_trims,
            dt,
            helicopter_model,
            helicopter_index,
            helicopter_env,
        )
        A.append(A_t)
        B.append(B_t)

    K, P = lqr_ltv(A, B, Q, R, Qfinal)

    return LinearController(K, P, trajectory, [hover_trims for _ in range(H)], time_invariant=False)


def test_tracking_controller_(
    tracking_controller,
    trajectory,
    helicopter_model,
    helicopter_index,
    helicopter_env,
    plot=True,
    early_stop=False,
    alpha=None,
    add_noise=True,
):
    H = trajectory.shape[0] - 1
    x_result = np.zeros((12, H + 1))
    u_result = np.zeros((4, H))

    x_result[:, 0] = trajectory[0, :]
    cost = 0.0
    for t in range(H):

        u_result[:, t] = tracking_controller.act(x_result[:, t], t, alpha=alpha)

        cost += cost_state(x_result[:, t], trajectory[t, :], Q)
        cost += cost_control(u_result[:, t], hover_trims, R)

        if early_stop and np.linalg.norm(x_result[:, t] - trajectory[t, :]) > 2:
            print("Stopping early at t:", t)
            return x_result[:, : t + 1], u_result[:, :t], cost

        # Simulate
        noise_F_T = get_tracking_noise() if add_noise else np.zeros(6)
        x_result[:, t + 1] = helicopter_env.step(
            x_result[:, t], u_result[:, t], dt, helicopter_model, helicopter_index, noise_F_T
        )

    cost += cost_final(x_result[:, H], trajectory[H, :], Qfinal)

    if plot:
        plt.subplot(3, 1, 1)
        ned_idx = helicopter_index.ned
        plt.plot(np.arange(H + 1), x_result[ned_idx[0], :], label="north", color="blue")
        plt.plot(np.arange(H + 1), trajectory[:, ned_idx[0]], "--", color="blue")
        plt.plot(np.arange(H + 1), x_result[ned_idx[1], :], label="east", color="red")
        plt.plot(np.arange(H + 1), trajectory[:, ned_idx[1]], "--", color="red")
        plt.plot(np.arange(H + 1), x_result[ned_idx[2], :], label="down", color="green")
        plt.plot(np.arange(H + 1), trajectory[:, ned_idx[2]], "--", color="green")
        plt.legend()

        plt.subplot(3, 1, 2)
        axis_angle_idx = helicopter_index.axis_angle
        plt.plot(np.arange(H + 1), x_result[axis_angle_idx[0], :], label="angle x", color="blue")
        plt.plot(np.arange(H + 1), trajectory[:, axis_angle_idx[0]], "--", color="blue")
        plt.plot(np.arange(H + 1), x_result[axis_angle_idx[1], :], label="angle y", color="red")
        plt.plot(np.arange(H + 1), trajectory[:, axis_angle_idx[1]], "--", color="red")
        plt.plot(np.arange(H + 1), x_result[axis_angle_idx[2], :], label="angle z", color="green")
        plt.plot(np.arange(H + 1), trajectory[:, axis_angle_idx[2]], "--", color="green")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(np.arange(H), u_result[0, :], label="aileron")
        plt.plot(np.arange(H), u_result[1, :], label="elevator")
        plt.plot(np.arange(H), u_result[2, :], label="rudder")
        plt.plot(np.arange(H), u_result[3, :], label="collective")
        plt.legend()

        plt.show()

    return x_result, u_result, cost


def test_tracking_controller(nose_in_funnel: bool):
    model, index, env = setup_env()
    trajectory = (
        desired_trajectory(index) if not nose_in_funnel else nose_in_funnel_trajectory(index)
    )
    controller = tracking_controller(trajectory, model, index, env)
    test_tracking_controller_(controller, trajectory, model, index, env)
