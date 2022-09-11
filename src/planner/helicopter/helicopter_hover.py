import warnings
from src.env.helicopter.helicopter_env import setup_env
from src.env.helicopter.helicopter_model import dt
from src.env.helicopter.linearized_helicopter_dynamics import linearized_heli_dynamics_2
from src.planner.lqr import lqr_lti
import numpy as np
import matplotlib.pyplot as plt

hover_at_zero = np.zeros(12)
hover_trims = np.zeros(4)


def hover_controller(helicopter_model, helicopter_index, helicopter_env):
    Q = np.diag(np.ones(13))
    R = np.eye(4)
    Qfinal = Q.copy()

    ## Design LQR controller
    ## Suggestion:
    # 1. Find A, B such that x_{t+1} = Ax_t + Bu_t
    # 2. Use provided Q and R matrices and loop over LQR backup to find steady state feedback matrix K

    A, B = linearized_heli_dynamics_2(
        hover_at_zero,
        hover_at_zero,
        hover_trims,
        dt,
        helicopter_model,
        helicopter_index,
        helicopter_env,
    )

    K = lqr_lti(A, B, Q, R)

    return K


def test_hover_controller_(
    hover_controller,
    helicopter_model,
    helicopter_index,
    helicopter_env,
    H,
    plot=True,
    early_stop=False,
):
    x_result = np.zeros((12, H + 1))
    x_result[:, 0] = hover_at_zero.copy()

    u_result = np.zeros((4, H))

    for t in range(H):
        # Control law; should fill in the control law
        # u_result[:, t] = np.random.randn(4)
        u_result[:, t] = hover_trims + hover_controller.dot(
            np.append(x_result[:, t] - hover_at_zero, 1)
        )

        if (
            early_stop
            and np.linalg.norm(
                np.concatenate([x_result[:, t] - hover_at_zero, u_result[:, t] - hover_trims])
            )
            > 5
        ):
            print("Stopping early at t:", t)
            return x_result[:, : t + 1], u_result[:, :t]

        # Simulate
        noise_F_t = np.random.randn(6)
        x_result[:, t + 1] = helicopter_env.step(
            x_result[:, t], u_result[:, t], dt, helicopter_model, helicopter_index, noise_F_t
        )

    print("Reached end of horizon", H)

    if plot:
        plt.subplot(3, 1, 1)
        ned_idx = helicopter_index.ned
        plt.plot(np.arange(H + 1), x_result[ned_idx[0], :], label="north")
        plt.plot(np.arange(H + 1), x_result[ned_idx[1], :], label="east")
        plt.plot(np.arange(H + 1), x_result[ned_idx[2], :], label="down")
        plt.legend()

        plt.subplot(3, 1, 2)
        axis_angle_idx = helicopter_index.axis_angle
        plt.plot(np.arange(H + 1), x_result[axis_angle_idx[0], :], label="angle x")
        plt.plot(np.arange(H + 1), x_result[axis_angle_idx[1], :], label="angle y")
        plt.plot(np.arange(H + 1), x_result[axis_angle_idx[2], :], label="angle z")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(np.arange(H), u_result[0, :], label="aileron")
        plt.plot(np.arange(H), u_result[1, :], label="elevator")
        plt.plot(np.arange(H), u_result[2, :], label="rudder")
        plt.plot(np.arange(H), u_result[3, :], label="collective")
        plt.legend()

        plt.show()

    return x_result, u_result


def test_hover_controller():
    model, index, env = setup_env()
    controller = hover_controller(model, index, env)
    test_hover_controller_(controller, model, index, env, H=300, plot=True)
