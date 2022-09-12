from time import time
from src.env.helicopter.helicopter_model import LinearizedHelicopterModel, dt
from src.env.helicopter.helicopter_env import HelicopterEnv, LinearizedHelicopterEnv, setup_env
from src.env.helicopter.linearized_helicopter_dynamics import linearized_heli_dynamics_2
from src.planner.helicopter.helicopter_hover import (
    hover_at_zero,
    hover_controller,
    hover_trims,
    test_hover_controller_,
)
from src.planner.lqr import lqr_lti

import numpy as np
from collections import deque
import torch


def initial_model():
    A = np.eye(13)
    B = np.eye(13, 4)

    return LinearizedHelicopterModel(A, B, time_varying=False)


def desired_trajectory_exploration_distribution(H, noise_state, noise_control):
    """
    Exploration distribution is encoded as a list of tuples of tuples
    where each element of the list contains ((s, n_s), (u, n_u)) where
    s, n_s represents the mean and std of the gaussian distribution for
    state, and u, n_u represents the mean and std of the gaussian
    distribution for control
    """
    exploration_distribution = []
    for t in range(H):
        exploration_distribution.append(
            ((hover_at_zero, noise_state), (hover_trims, noise_control))
        )
    return exploration_distribution


def expert_exploration_distribution(
    helicopter_env, helicopter_model, helicopter_index, H, noise_state, noise_control
):
    expert_controller = hover_controller(helicopter_model, helicopter_index, helicopter_env)
    exploration_distribution = []
    state = hover_at_zero.copy()
    for t in range(H):
        control = hover_trims + expert_controller.dot(np.append(state - hover_at_zero, 1))
        exploration_distribution.append(((state, noise_state), (control, noise_control)))
        # TODO: Should I be adding noise when rolling out expert controller?
        state = helicopter_env.step(
            state, control, dt, helicopter_model, helicopter_index, noise=np.random.randn(6)
        )

    return exploration_distribution


def optimal_controller(linearized_helicopter_model):
    Q = np.diag(np.ones(13))
    R = np.eye(4)
    Qfinal = Q.copy()

    K = lqr_lti(linearized_helicopter_model.A, linearized_helicopter_model.B, Q, R)

    return K


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


def agnostic_sys_id_hover_learner_(
    helicopter_env,
    helicopter_model,
    helicopter_index,
    num_iterations=50,
    num_samples_per_iteration=100,
    exploration_distribution_type="desired_trajectory",
):
    nominal_model = initial_model()
    model = initial_model()
    # A, B = linearized_heli_dynamics_2(
    #     hover_at_zero,
    #     hover_at_zero,
    #     hover_trims,
    #     dt,
    #     helicopter_model,
    #     helicopter_index,
    #     helicopter_env,
    # )
    # nominal_model = LinearizedHelicopterModel(A, B, time_varying=False)
    # model = LinearizedHelicopterModel(A, B, time_varying=False)
    controller = optimal_controller(model)
    H = 400
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

    for n in range(num_iterations):
        # Rollout controller in real world
        x_result, u_result = test_hover_controller_(
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
                # Get mean, std from exploration distribution
                (mean_state, std_state), (mean_control, std_control) = exploration_distribution[t]
                # Sample state and control
                state, control = np.random.normal(mean_state, std_state), np.random.normal(
                    mean_control, std_control
                )
                # Get next state from env
                next_state = helicopter_env.step(
                    state, control, dt, helicopter_model, helicopter_index, noise=np.random.randn(6)
                )
                # Add to dataset
                dataset.append((state, control, next_state))
            else:
                ## Sample from current policy
                # Sample a random timestamp
                t = np.random.randint(u_result.shape[1])
                # Get state, control, and next state from current policy
                state, control, next_state = x_result[:, t], u_result[:, t], x_result[:, t + 1]
                # Add to dataset
                dataset.append((state, control, next_state))

        # Fit new model
        model = fit_model(dataset, nominal_model)

        # Compute new optimal controller
        controller = optimal_controller(model)

    return controller


def agnostic_sys_id_hover_learner():
    np.random.seed(0)
    model, index, env = setup_env()
    agnostic_sys_id_hover_learner_(env, model, index)
