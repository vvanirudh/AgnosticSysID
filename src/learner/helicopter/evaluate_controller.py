from cmath import isnan
import numpy as np
import warnings
from src.env.helicopter.helicopter_env import HelicopterEnv
from src.env.helicopter.helicopter_model import HelicopterIndex, dt
from src.env.helicopter.linearized_helicopter_dynamics import (
    linearized_heli_dynamics,
    linearized_heli_dynamics_2,
)
from src.planner.helicopter.helicopter_hover import (
    test_hover_controller_,
    hover_at_zero,
    hover_trims,
    Q,
    R,
    Qfinal,
    hover_controller,
)
from src.planner.helicopter.helicopter_track_trajectory import (
    test_tracking_controller_,
    tracking_controller,
    Q_track,
    R_track,
    Qfinal_track,
)
from src.planner.lqr import (
    lqr_linearized_tv,
    lqr_linearized_tv_2,
    lqr_linearized_tv_3,
    lqr_lti,
    lqr_ltv,
)
from src.planner.helicopter.controller import (
    LinearController,
    LinearControllerWithFeedForward,
    LinearControllerWithFeedForwardAndNoOffset,
    ManualController,
)


def evaluate_hover_controller(
    controller,
    x_target,
    u_target,
    helicopter_model,
    helicopter_index,
    helicopter_env,
    H,
    add_noise=True,
):
    _, _, cost = test_hover_controller_(
        controller,
        helicopter_model,
        helicopter_index,
        helicopter_env,
        H,
        plot=False,
        early_stop=False,
        add_noise=add_noise,
    )
    return cost


def evaluate_tracking_controller(
    controller, trajectory, helicopter_model, helicopter_index, helicopter_env, add_noise=True
):
    _, _, cost = test_tracking_controller_(
        controller,
        trajectory,
        helicopter_model,
        helicopter_index,
        helicopter_env,
        plot=False,
        early_stop=False,
        add_noise=add_noise,
    )
    return cost


def zero_hover_controller(model):
    return LinearController(
        np.zeros((4, 13)), np.zeros((13, 13)), hover_at_zero, hover_trims, time_invariant=True
    )


def random_hover_controller(model):
    K = np.random.randn(4, 13) * 1e-7
    P = np.random.randn(13, 13)
    return LinearController(K, P, hover_at_zero, hover_trims, time_invariant=True)


def optimal_hover_controller_for_linearized_model(model):
    K, P = lqr_lti(model.A, model.B, Q, R)

    return LinearController(K, P, hover_at_zero, hover_trims, time_invariant=True)


def optimal_tracking_controller_for_linearized_model(model, trajectory):
    H = trajectory.shape[0] - 1
    K, P = lqr_ltv(model.A, model.B, Q_track, R_track, Qfinal_track)

    return LinearController(K, P, trajectory, [hover_trims for _ in range(H)], time_invariant=False)


def optimal_hover_controller_for_parameterized_model_psdp(model):
    helicopter_env = HelicopterEnv()
    helicopter_index = HelicopterIndex()
    return hover_controller(model, helicopter_index, helicopter_env)


def optimal_tracking_controller_for_parameterized_model_psdp(model, trajectory):
    helicopter_env = HelicopterEnv()
    helicopter_index = HelicopterIndex()
    return tracking_controller(trajectory, model, helicopter_index, helicopter_env)


def optimal_hover_controller_for_parameterized_model_ilqr(model, H):
    helicopter_env = HelicopterEnv()
    helicopter_index = HelicopterIndex()
    test_controller_fn = lambda controller, alpha: test_hover_controller_(
        controller,
        model,
        helicopter_index,
        helicopter_env,
        H,
        plot=False,
        early_stop=False,
        add_noise=False,
        alpha=alpha,
    )
    target_states = np.array([hover_at_zero.copy() for _ in range(H + 1)])
    target_controls = np.array([hover_trims.copy() for _ in range(H)])
    initial_controller = hover_controller(model, helicopter_index, helicopter_env)

    return optimal_controller_for_parameterized_model_ilqr(
        model, H, initial_controller, test_controller_fn, target_states, target_controls
    )


def optimal_tracking_controller_for_parameterized_model_ilqr(model, trajectory):
    helicopter_env = HelicopterEnv()
    helicopter_index = HelicopterIndex()
    H = trajectory.shape[0] - 1
    test_controller_fn = lambda controller, alpha: test_tracking_controller_(
        controller,
        trajectory,
        model,
        helicopter_index,
        helicopter_env,
        plot=False,
        early_stop=False,
        add_noise=False,
        alpha=alpha,
    )
    target_controls = np.array([hover_trims.copy() for _ in range(H)])
    initial_controller = tracking_controller(trajectory, model, helicopter_index, helicopter_env)

    return optimal_controller_for_parameterized_model_ilqr(
        model, H, initial_controller, test_controller_fn, trajectory, target_controls
    )


def optimal_controller_for_parameterized_model_ilqr(
    model, H, initial_controller, test_controller_fn, target_states, target_controls
):
    controller = initial_controller

    x_result, u_result, cost = test_controller_fn(controller, 1.0)
    print("ILQR", cost)
    NUM_ILQR_ITERATIONS = 1000
    for _ in range(NUM_ILQR_ITERATIONS):
        # Linearize dynamics and quadraticize cost around trajectory
        # TODO: Can parallelize the code below
        # A, B, C_x, C_u, C_xx, C_uu = [], [], [], [], [], []
        # for t in range(H):
        #     A_t, B_t = linearized_heli_dynamics_2(
        #         x_result[:, t],
        #         x_result[:, t + 1],
        #         u_result[:, t],
        #         dt,
        #         model,
        #         helicopter_index,
        #         helicopter_env,
        #         offset=False,
        #     )
        #     C_x_t = Q[:12, :12] @ (x_result[:, t] - target_states[t, :])
        #     C_u_t = R @ (u_result[:, t] - target_controls[t, :])
        #     A.append(A_t)
        #     B.append(B_t)
        #     C_x.append(C_x_t)
        #     C_u.append(C_u_t)
        #     C_xx.append(Q[:12, :12])
        #     C_uu.append(R)
        result = [
            linearize_dynamics_and_quadraticize_cost(
                x_result[:, t],
                x_result[:, t + 1],
                u_result[:, t],
                target_states[t, :],
                target_controls[t, :],
                model,
            )
            for t in range(H)
        ]
        A, B, C_x, C_xx, C_u, C_uu = list(zip(*result))
        C_x_f = Qfinal[:12, :12] @ (x_result[:, H] - target_states[H, :])
        C_xx_f = Qfinal[:12, :12]
        # Run LQR
        try:
            k, K = lqr_linearized_tv(A, B, C_x, C_u, C_xx, C_uu, C_x_f, C_xx_f)
            new_controller = LinearControllerWithFeedForwardAndNoOffset(
                k, K, x_result.T, u_result.T, time_invariant=False
            )
        except np.linalg.LinAlgError as err:
            print(err)
            new_controller = controller

        # Rollout controller in the model to get trajectory
        alpha_found = False
        alpha = 1.0
        for _ in range(20):
            new_x_result, new_u_result, new_cost = test_controller_fn(new_controller, alpha)
            # print("\t", new_cost, alpha)
            if new_cost < cost:
                controller = ManualController(new_u_result.T)
                x_result = new_x_result.copy()
                u_result = new_u_result.copy()
                cost = new_cost
                alpha_found = True
                break
            alpha = 0.5 * alpha
        if not alpha_found:
            break
        print("ILQR", cost, alpha)

    return controller


def linearize_dynamics_and_quadraticize_cost(
    state, next_state, control, target_state, target_control, model
):
    helicopter_index = HelicopterIndex()
    helicopter_env = HelicopterEnv()
    A_t, B_t = linearized_heli_dynamics_2(
        state,
        next_state,
        control,
        dt,
        model,
        helicopter_index,
        helicopter_env,
        offset=False,
    )
    C_x_t = Q[:12, :12] @ (state - target_state)
    C_u_t = R @ (control - target_control)
    C_xx_t = Q[:12, :12]
    C_uu_t = R.copy()
    return A_t, B_t, C_x_t, C_xx_t, C_u_t, C_uu_t


###################### DEPRECATED #####################


def optimal_hover_ilqr_controller_for_parameterized_model(model, H, controller=None):
    warnings.warn("DEPRECATED! Use optimal_controller_for_parameterized_model_ilqr")
    helicopter_env = HelicopterEnv()
    helicopter_index = HelicopterIndex()
    if controller is None:
        # controller = hover_controller(model, helicopter_index, helicopter_env)
        # controller = zero_hover_controller(model)
        controller = random_hover_controller(model)

    x_result, u_result, cost = test_hover_controller_(
        controller,
        model,
        helicopter_index,
        helicopter_env,
        H,
        plot=False,
        early_stop=False,
        add_noise=False,
    )
    print(cost)

    for _ in range(1000):
        # Linearize dynamics and quadraticize cost around trajectory
        # TODO: Can parallelize the code below
        A, B, C_x, C_u, C_xx, C_uu, residuals = [], [], [], [], [], [], []
        for t in range(H):
            A_t, B_t = linearized_heli_dynamics_2(
                x_result[:, t],
                x_result[:, t + 1],
                u_result[:, t],
                dt,
                model,
                helicopter_index,
                helicopter_env,
            )
            C_x_t = Q @ (np.append(x_result[:, t] - hover_at_zero, 1))
            C_u_t = R @ (u_result[:, t] - hover_trims)
            A.append(A_t)
            B.append(B_t)
            C_x.append(C_x_t)
            C_u.append(C_u_t)
            C_xx.append(Q)
            C_uu.append(R)
            residuals.append(
                np.append(x_result[:, t + 1], 1)
                - A_t @ np.append(x_result[:, t], 1)
                - B_t @ u_result[:, t]
            )
        C_x_f = Qfinal @ (np.append(x_result[:, H] - hover_at_zero, 1))
        C_xx_f = Qfinal
        # Run LQR
        try:
            # k, K = lqr_linearized_tv(A, B, C_x, C_u, C_xx, C_uu)
            k, K = lqr_linearized_tv_2(A, B, C_x, C_u, C_xx, C_uu, C_x_f, C_xx_f, residuals)
            # k, K = lqr_linearized_tv_3(A, B, C_x, C_u, C_xx, C_uu, C_x_f, C_xx_f)
            new_controller = LinearControllerWithFeedForward(
                k, K, x_result.T, u_result.T, time_invariant=False
            )
        except np.linalg.LinAlgError as err:
            print(err)
            new_controller = controller

        # Rollout controller in the model to get trajectory
        alpha_found = False
        alpha = 1.0
        for _ in range(10):
            new_x_result, new_u_result, new_cost = test_hover_controller_(
                new_controller,
                model,
                helicopter_index,
                helicopter_env,
                H,
                plot=False,
                early_stop=False,
                alpha=alpha,
                add_noise=False,
            )
            print("\t", new_cost, alpha)
            if new_cost < cost:
                controller = ManualController(new_u_result.T)
                x_result = new_x_result.copy()
                u_result = new_u_result.copy()
                cost = new_cost
                alpha_found = True
                break
            alpha = 0.5 * alpha
        if not alpha_found:
            break
        print(cost, alpha)

    return controller


def optimal_tracking_ilqr_controller_for_parameterized_model(model, trajectory, controller=None):
    warnings.warn("DEPRECATED! Use optimal_controller_for_parameterized_model_ilqr")
    H = trajectory.shape[0] - 1
    helicopter_env = HelicopterEnv()
    helicopter_index = HelicopterIndex()
    if controller is None:
        controller = tracking_controller(trajectory, model, helicopter_index, helicopter_env)

    x_result, u_result, cost = test_tracking_controller_(
        controller,
        trajectory,
        model,
        helicopter_index,
        helicopter_env,
        plot=False,
        early_stop=False,
        add_noise=False,
    )
    print(cost)

    for _ in range(100):
        # Linearize dynamics and quadraticize cost around trajectory
        A, B, C_x, C_u, C_xx, C_uu, residuals = [], [], [], [], [], [], []
        for t in range(H):
            A_t, B_t = linearized_heli_dynamics_2(
                x_result[:, t],
                x_result[:, t + 1],
                u_result[:, t],
                dt,
                model,
                helicopter_index,
                helicopter_env,
            )
            C_x_t = Q_track @ (np.append(x_result[:, t] - trajectory[t, :], 1))
            C_u_t = R_track @ (u_result[:, t] - hover_trims)
            A.append(A_t)
            B.append(B_t)
            C_x.append(C_x_t)
            C_u.append(C_u_t)
            C_xx.append(Q_track)
            C_uu.append(R_track)
            residuals.append(
                np.append(x_result[:, t + 1], 1)
                - A_t @ np.append(x_result[:, t], 1)
                - B_t @ u_result[:, t]
            )
        C_x_f = Qfinal_track @ (np.append(x_result[:, H] - trajectory[H, :], 1))
        C_xx_f = Qfinal_track
        try:
            # k, K = lqr_linearized_tv(A, B, C_x, C_u, C_xx, C_uu)
            k, K = lqr_linearized_tv_2(A, B, C_x, C_u, C_xx, C_uu, C_x_f, C_xx_f, residuals)
            # k, K = lqr_linearized_tv_3(A, B, C_x, C_u, C_xx, C_uu, C_x_f, C_xx_f)
            new_controller = LinearControllerWithFeedForward(
                k, K, x_result.T, u_result.T, time_invariant=False
            )
        except np.linalg.LinAlgError as err:
            print(err)
            new_controller = controller
        # Rollout controller in the model to get trajectory
        alpha_found = False
        alpha = 1.0
        for _ in range(100):
            new_x_result, new_u_result, new_cost = test_tracking_controller_(
                new_controller,
                trajectory,
                model,
                helicopter_index,
                helicopter_env,
                plot=False,
                early_stop=False,
                alpha=alpha,
                add_noise=False,
            )
            print("\t", new_cost, alpha)
            if new_cost < cost:
                controller = ManualController(new_u_result.T)
                x_result = new_x_result.copy()
                u_result = new_u_result.copy()
                cost = new_cost
                alpha_found = True
                break
            alpha = 0.5 * alpha
        if not alpha_found:
            break
        print(cost, alpha)

    return controller


def optimal_hover_ilqr_controller_for_parameterized_model_2(model, H, controller=None):
    warnings.warn("DEPRECATED! Use optimal_controller_for_parameterized_model_ilqr")
    helicopter_env = HelicopterEnv()
    helicopter_index = HelicopterIndex()
    if controller is None:
        controller = hover_controller(model, helicopter_index, helicopter_env)
        # controller = zero_hover_controller(model)
        # controller = random_hover_controller(model)

    x_result, u_result, cost = test_hover_controller_(
        controller,
        model,
        helicopter_index,
        helicopter_env,
        H,
        plot=False,
        early_stop=False,
        add_noise=False,
    )
    print(cost)

    for _ in range(1000):
        # Linearize dynamics and quadraticize cost around trajectory
        # TODO: Can parallelize the code below
        A, B, C_x, C_u, C_xx, C_uu, residuals = [], [], [], [], [], [], []
        for t in range(H):
            A_t, B_t = linearized_heli_dynamics_2(
                x_result[:, t],
                x_result[:, t + 1],
                u_result[:, t],
                dt,
                model,
                helicopter_index,
                helicopter_env,
                offset=False,
            )
            C_x_t = Q[:12, :12] @ (x_result[:, t] - hover_at_zero)
            C_u_t = R @ (u_result[:, t] - hover_trims)
            A.append(A_t)
            B.append(B_t)
            C_x.append(C_x_t)
            C_u.append(C_u_t)
            C_xx.append(Q[:12, :12])
            C_uu.append(R)
            residuals.append(x_result[:, t + 1] - A_t @ x_result[:, t] - B_t @ u_result[:, t])
        C_x_f = Qfinal[:12, :12] @ (x_result[:, H] - hover_at_zero)
        C_xx_f = Qfinal[:12, :12]
        # Run LQR
        try:
            k, K = lqr_linearized_tv(A, B, C_x, C_u, C_xx, C_uu, C_x_f, C_xx_f)
            # k, K = lqr_linearized_tv_2(A, B, C_x, C_u, C_xx, C_uu, C_x_f, C_xx_f, residuals)
            # k, K = lqr_linearized_tv_3(A, B, C_x, C_u, C_xx, C_uu, C_x_f, C_xx_f)
            new_controller = LinearControllerWithFeedForwardAndNoOffset(
                k, K, x_result.T, u_result.T, time_invariant=False
            )
        except np.linalg.LinAlgError as err:
            print(err)
            new_controller = controller

        # Rollout controller in the model to get trajectory
        alpha_found = False
        alpha = 1.0
        for _ in range(100):
            new_x_result, new_u_result, new_cost = test_hover_controller_(
                new_controller,
                model,
                helicopter_index,
                helicopter_env,
                H,
                plot=False,
                early_stop=False,
                alpha=alpha,
                add_noise=False,
            )
            # print("\t", new_cost, alpha)
            if new_cost < cost:
                controller = ManualController(new_u_result.T)
                x_result = new_x_result.copy()
                u_result = new_u_result.copy()
                cost = new_cost
                alpha_found = True
                break
            alpha = 0.5 * alpha
        if not alpha_found:
            break
        print(cost, alpha)

    return controller
