from src.env.helicopter.helicopter_model import dt
from src.env.helicopter.helicopter_env import setup_env
from src.planner.helicopter.helicopter_hover import (
    hover_at_zero,
    hover_controller,
    hover_trims,
    test_hover_controller_,
)
from src.learner.helicopter.exploration_distribution import (
    desired_trajectory_exploration_distribution,
    expert_exploration_distribution,
)
from src.learner.helicopter.fit_model import (
    fit_linearized_model,
    fit_parameterized_model,
    initial_linearized_model,
    initial_parameterized_model,
)
from src.learner.helicopter.evaluate_controller import (
    optimal_controller_for_linearized_model,
    evaluate_controller,
    optimal_controller_for_parameterized_model,
    optimal_ilqr_controller_for_parameterized_model,
)

import numpy as np
import ray
from collections import deque
import matplotlib.pyplot as plt

WARM_START_ITERATION = 10


def agnostic_sys_id_hover_learner_(
    helicopter_env,
    helicopter_model,
    helicopter_index,
    linearized_model: bool,
    num_iterations=100,
    num_samples_per_iteration=100,
    exploration_distribution_type="desired_trajectory",
):
    H = 100
    nominal_model = (
        initial_linearized_model(H) if linearized_model else initial_parameterized_model()
    )
    model = initial_linearized_model(H) if linearized_model else initial_parameterized_model()
    controller = (
        optimal_controller_for_linearized_model(model)
        if linearized_model
        else optimal_controller_for_parameterized_model(model)
    )
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
            add_noise=True,
        )

        for k in range(num_samples_per_iteration):
            toss = np.random.rand()
            # Check if controller is very very bad
            if toss < 0.5 or u_result.shape[1] == 0:
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
        model = (
            fit_linearized_model(dataset, nominal_model)
            if linearized_model
            else fit_parameterized_model(dataset, nominal_model)
        )

        # Compute new optimal controller
        controller = (
            optimal_controller_for_linearized_model(model)
            if linearized_model
            else optimal_ilqr_controller_for_parameterized_model(model, H)
        )

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


def agnostic_sys_id_hover_learner(linearized_model: bool):
    np.random.seed(0)
    # if not linearized_model:
    #     ray.init()
    model, index, env = setup_env()
    agnostic_sys_id_hover_learner_(env, model, index, linearized_model)
