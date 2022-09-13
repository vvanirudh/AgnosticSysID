from cmath import isnan
from time import time
from src.env.helicopter.helicopter_model import LinearizedHelicopterModel, dt
from src.env.helicopter.helicopter_env import HelicopterEnv, LinearizedHelicopterEnv, setup_env
from src.env.helicopter.linearized_helicopter_dynamics import linearized_heli_dynamics_2
from src.planner.helicopter.helicopter_hover import (
    hover_at_zero,
    hover_controller,
    hover_trims,
    test_hover_controller_,
    Q,
    R,
    Qfinal,
)
from src.planner.lqr import lqr_lti
from src.planner.helicopter.controller import LinearController
from src.learner.helicopter.exploration_distribution import (
    desired_trajectory_exploration_distribution,
    expert_exploration_distribution,
)

import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt


def evaluate_controller(
    controller, x_target, u_target, helicopter_model, helicopter_index, helicopter_env, H
):
    _, _, cost = test_hover_controller_(
        controller,
        helicopter_model,
        helicopter_index,
        helicopter_env,
        H,
        plot=False,
        early_stop=False,
    )
    if isnan(cost):
        return 1e5
    return min(cost, 1e5)


def initial_model(H, time_varying=False):
    A = np.eye(13) if not time_varying else [np.eye(13) for _ in range(H)]
    B = np.eye(13, 4) if not time_varying else [np.eye(13, 4) for _ in range(H)]

    return LinearizedHelicopterModel(A, B, time_varying=time_varying)


def optimal_controller(linearized_helicopter_model):
    K, P = lqr_lti(linearized_helicopter_model.A, linearized_helicopter_model.B, Q, R)

    return LinearController(K, P, hover_at_zero, hover_trims, time_invariant=True)


def fit_model(dataset, nominal_model):
    # Construct training data
    states, controls, next_states = tuple(zip(*dataset))
    states, controls, next_states = (
        torch.from_numpy(np.array(states)),
        torch.from_numpy(np.array(controls)),
        torch.from_numpy(np.array(next_states)),
    )
    hover_at_zero_ = torch.from_numpy(hover_at_zero.reshape(1, -1))
    states, next_states = states - hover_at_zero_, next_states - hover_at_zero_
    states, next_states = torch.cat([states, torch.ones(states.shape[0], 1)], dim=1), torch.cat(
        [next_states, torch.ones(states.shape[0], 1)], dim=1
    )

    nominal_A = torch.from_numpy(nominal_model.A)
    nominal_B = torch.from_numpy(nominal_model.B)
    A_fit = torch.zeros_like(nominal_A, requires_grad=True)
    B_fit = torch.zeros_like(nominal_B, requires_grad=True)
    optimizer = torch.optim.Adam([A_fit, B_fit])

    loss = np.inf
    loss_old = -np.inf
    i = 0
    while abs(loss_old - loss) > 1e-8:
        loss_old = loss
        optimizer.zero_grad()
        loss = torch.mean(
            torch.norm(
                torch.matmul(states, (nominal_A + A_fit).T)
                + torch.matmul(controls, (nominal_B + B_fit).T)
                - next_states,
                dim=1,
            )
        )
        # Add regularization
        loss += (1e-3 / np.sqrt(len(dataset))) * (torch.norm(A_fit) ** 2 + torch.norm(B_fit) ** 2)
        # print("Iteration: ", i, "Loss: ", loss)
        # Backprop and update
        loss.backward()
        optimizer.step()
        i += 1

    return LinearizedHelicopterModel(
        nominal_model.A + A_fit.detach().numpy(),
        nominal_model.B + B_fit.detach().numpy(),
        time_varying=False,
    )


def fit_model_numpy(dataset, nominal_model):
    states, controls, next_states = tuple(zip(*dataset))
    states, controls, next_states = np.array(states), np.array(controls), np.array(next_states)
    hover_at_zero_ = hover_at_zero.reshape(1, -1)
    hover_trims_ = hover_trims.reshape(1, -1)
    states, controls, next_states = (
        states - hover_at_zero_,
        controls - hover_trims_,
        next_states - hover_at_zero_,
    )
    states, next_states = np.hstack([states, np.ones((states.shape[0], 1))]), np.hstack(
        [next_states, np.ones((next_states.shape[0], 1))]
    )

    next_states = next_states - states.dot(nominal_model.A.T) - controls.dot(nominal_model.B.T)

    states_controls, next_states_zeros = np.hstack([states, controls]), np.hstack(
        [next_states, np.zeros_like(controls)]
    )

    solution = np.linalg.lstsq(states_controls, next_states_zeros, rcond=None)[0]
    A_fit = solution[:13, :13].T
    B_fit = solution[13:, :13].T
    return LinearizedHelicopterModel(
        nominal_model.A + A_fit, nominal_model.B + B_fit, time_varying=False
    )


def fit_value_aware_model(dataset, Ps, nominal_model):
    # Construct training data
    states, controls, next_states = tuple(zip(*dataset))
    states, controls, next_states = (
        torch.from_numpy(np.array(states)),
        torch.from_numpy(np.array(controls)),
        torch.from_numpy(np.array(next_states)),
    )
    hover_at_zero_ = torch.from_numpy(hover_at_zero.reshape(1, -1))
    states, next_states = states - hover_at_zero_, next_states - hover_at_zero_
    states, next_states = torch.cat([states, torch.ones(states.shape[0], 1)], dim=1), torch.cat(
        [next_states, torch.ones(states.shape[0], 1)], dim=1
    )

    nominal_A = torch.from_numpy(nominal_model.A)
    nominal_B = torch.from_numpy(nominal_model.B)
    A_fit = torch.zeros_like(nominal_A, requires_grad=True)
    B_fit = torch.zeros_like(nominal_B, requires_grad=True)
    optimizer = torch.optim.Adam([A_fit, B_fit])

    loss = np.inf
    loss_old = -np.inf
    i = 0
    while abs(loss_old - loss) > 1e-8:
        loss_old = loss
        optimizer.zero_grad()
        loss = torch.mean(
            torch.norm(
                torch.matmul(states, (nominal_A + A_fit).T)
                + torch.matmul(controls, (nominal_B + B_fit).T)
                - next_states,
                dim=1,
            )
        )
        # Add regularization
        loss += (1e-3 / np.sqrt(len(dataset))) * (torch.norm(A_fit) ** 2 + torch.norm(B_fit) ** 2)
        # print("Iteration: ", i, "Loss: ", loss)
        # Backprop and update
        loss.backward()
        optimizer.step()
        i += 1

    return LinearizedHelicopterModel(
        nominal_model.A + A_fit.detach().numpy(),
        nominal_model.B + B_fit.detach().numpy(),
        time_varying=False,
    )


def agnostic_sys_id_hover_learner_(
    helicopter_env,
    helicopter_model,
    helicopter_index,
    num_iterations=100,
    num_samples_per_iteration=100,
    exploration_distribution_type="desired_trajectory",
):
    H = 400
    nominal_model = initial_model(H)
    model = initial_model(H)
    controller = optimal_controller(model)
    dataset = deque(maxlen=10000)

    if exploration_distribution_type == "desired_trajectory":
        exploration_distribution = desired_trajectory_exploration_distribution(H, 0.0025, 0.0001)
    elif exploration_distribution_type == "expert_controller":
        exploration_distribution = expert_exploration_distribution(
            helicopter_env, helicopter_model, helicopter_index, H, 0.0, 0.0
        )
    elif exploration_distribution_type == "expert_controller_with_noise":
        exploration_distribution = expert_exploration_distribution(
            helicopter_env, helicopter_model, helicopter_index, H, 0.0, 0.0001
        )
    else:
        raise NotImplementedError("Unknown exploration distribution type")

    costs = []
    x_target = np.array([hover_at_zero for _ in range(H + 1)]).T
    u_target = np.array([hover_trims for _ in range(H)]).T
    # Evaluate controller
    costs.append(
        evaluate_controller(
            controller, x_target, u_target, helicopter_model, helicopter_index, helicopter_env, H
        )
    )

    for n in range(num_iterations):
        # Rollout controller in real world
        x_result, u_result, _ = test_hover_controller_(
            controller,
            helicopter_model,
            helicopter_index,
            helicopter_env,
            H,
            plot=False,
            early_stop=True,
        )

        for k in range(num_samples_per_iteration):
            toss = np.random.rand()
            if toss < 0.5:
                ## Sample from exploration distribution
                # Sample a random timestamp
                t = np.random.randint(H)
                # Sample state and control
                state, control = exploration_distribution.sample(t)
                # Get next state from env
                next_state = helicopter_env.step(
                    state, control, dt, helicopter_model, helicopter_index, noise=np.random.randn(6)
                )
            else:
                ## Sample from current policy
                # Sample a random timestamp
                t = np.random.randint(u_result.shape[1])
                # Get state, control, and next state from current policy
                state, control, next_state = x_result[:, t], u_result[:, t], x_result[:, t + 1]

            # Add to dataset
            dataset.append((state, control, next_state))

        # Fit new model
        # model = fit_model(dataset, nominal_model)
        model = fit_model_numpy(dataset, nominal_model)

        # Compute new optimal controller
        controller = optimal_controller(model)

        # Evaluate controller
        costs.append(
            evaluate_controller(
                controller,
                x_target,
                u_target,
                helicopter_model,
                helicopter_index,
                helicopter_env,
                H,
            )
        )

    best_controller = hover_controller(helicopter_model, helicopter_index, helicopter_env)
    best_cost = evaluate_controller(
        best_controller, x_target, u_target, helicopter_model, helicopter_index, helicopter_env, H
    )

    plt.plot(np.arange(num_iterations + 1), costs, label="DAgger")
    plt.plot(
        np.arange(num_iterations + 1),
        [best_cost for _ in range(num_iterations + 1)],
        "--",
        label="Opt",
    )
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.yscale("log")
    plt.show()

    return controller


def agnostic_sys_id_hover_learner():
    np.random.seed(0)
    model, index, env = setup_env()
    agnostic_sys_id_hover_learner_(env, model, index)
