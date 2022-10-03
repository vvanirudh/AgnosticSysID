from collections import deque
import numpy as np
import time
import matplotlib.pyplot as plt

from src.env.helicopter.helicopter_model import HelicopterIndex, dt
from src.env.helicopter.helicopter_env import HelicopterEnv, setup_env
from src.learner.helicopter.evaluate_controller import (
    evaluate_tracking_controller,
    optimal_tracking_controller_for_linearized_model,
    optimal_tracking_controller_for_parameterized_model,
    optimal_tracking_ilqr_controller_for_parameterized_model,
    random_hover_controller,
)
from src.learner.helicopter.exploration_distribution import (
    desired_tracking_trajectory_exploration_distribution,
    expert_tracking_exploration_distribution,
)
from src.learner.helicopter.fit_model import (
    fit_linearized_model,
    fit_linearized_time_varying_model,
    fit_parameterized_model,
    initial_linearized_model,
    initial_linearized_model_about_hover,
    initial_parameterized_model,
)
from src.learner.helicopter.noise import get_tracking_noise
from src.planner.helicopter.helicopter_track_trajectory import (
    desired_trajectory,
    nose_in_funnel_trajectory,
    test_tracking_controller_,
    tracking_controller,
)


def get_optimal_tracking_controller(model, trajectory, linearized_model: bool, pdl: bool):
    if linearized_model:
        controller = optimal_tracking_controller_for_linearized_model(model, trajectory)
    elif pdl:
        controller = optimal_tracking_controller_for_parameterized_model(model, trajectory)
    else:
        controller = optimal_tracking_ilqr_controller_for_parameterized_model(model, trajectory)
    return controller


def get_initial_tracking_model(H, linearized_model: bool):
    return (
        initial_linearized_model_about_hover(H, time_varying=True)
        if linearized_model
        else initial_parameterized_model()
    )


def agnostic_sys_id_tracking_learner_(
    helicopter_env: HelicopterEnv,
    helicopter_model,
    helicopter_index: HelicopterIndex,
    linearized_model: bool,
    pdl: bool,
    num_iterations=100,
    num_samples_per_iteration=500,
    exploration_distribution_type="expert_controller",
    plot=True,
    add_noise=True,
):
    trajectory = desired_trajectory(helicopter_index)
    # trajectory = nose_in_funnel_trajectory(helicopter_index)
    H = trajectory.shape[0] - 1
    nominal_model = get_initial_tracking_model(H, linearized_model)
    model = get_initial_tracking_model(H, linearized_model)
    controller = get_optimal_tracking_controller(model, trajectory, linearized_model, pdl)
    dataset = [deque(maxlen=10000) for _ in range(H)]

    if exploration_distribution_type == "desired_trajectory":
        exploration_distribution = desired_tracking_trajectory_exploration_distribution(
            trajectory, 0.0025, 0.0001
        )
    elif exploration_distribution_type == "expert_controller":
        exploration_distribution = expert_tracking_exploration_distribution(
            trajectory,
            helicopter_env,
            helicopter_model,
            helicopter_index,
            0.0,
            0.0,
            add_noise=add_noise,
        )
    elif exploration_distribution_type == "expert_controller_with_noise":
        exploration_distribution = expert_tracking_exploration_distribution(
            trajectory,
            helicopter_env,
            helicopter_model,
            helicopter_index,
            0.0,
            0.0001,
            add_noise=add_noise,
        )
    else:
        raise NotImplementedError("Unknown exploration distribution type")

    costs = []

    # Evaluate controller
    costs.append(
        evaluate_tracking_controller(
            controller,
            trajectory,
            helicopter_model,
            helicopter_index,
            helicopter_env,
            add_noise=False,
        )
    )
    print(costs[-1])
    total_time = 0.0
    for n in range(num_iterations):
        print("Iteration", n)
        # Rollout controller in real world
        x_result, u_result, _ = test_tracking_controller_(
            controller,
            trajectory,
            helicopter_model,
            helicopter_index,
            helicopter_env,
            plot=False,
            early_stop=True,
            add_noise=add_noise,
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
                    state,
                    control,
                    dt,
                    helicopter_model,
                    helicopter_index,
                    noise=get_tracking_noise() if add_noise else np.zeros(6),
                )
            else:
                ## Sample from current policy
                # Sample a random timestamp
                t = np.random.randint(u_result.shape[1])
                # Get state, control, and next state from current policy
                state, control, next_state = x_result[:, t], u_result[:, t], x_result[:, t + 1]

            # Add to dataset
            dataset[t].append((state, control, next_state))

        # Fit new model
        model = (
            fit_linearized_time_varying_model(dataset, nominal_model, trajectory)
            if linearized_model
            else fit_parameterized_model(dataset, nominal_model, previous_model=model)
        )

        # Compute new optimal controller
        start = time.time()
        controller = get_optimal_tracking_controller(model, trajectory, linearized_model, pdl)
        end = time.time()

        # Evaluate controller
        costs.append(
            evaluate_tracking_controller(
                controller,
                trajectory,
                helicopter_model,
                helicopter_index,
                helicopter_env,
                add_noise=False,
            )
        )
        print(costs[-1])

        total_time += end - start

    avg_time = total_time / num_iterations
    best_controller = optimal_tracking_ilqr_controller_for_parameterized_model(
        helicopter_model, trajectory
    )
    best_cost = evaluate_tracking_controller(
        best_controller,
        trajectory,
        helicopter_model,
        helicopter_index,
        helicopter_env,
        add_noise=False,
    )

    if plot:
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
        plt.title("Average planning time per iteration: " + str(avg_time))
        exp_name = "agnostic_sysid" if not pdl else "agnostic_sysid_pdl"
        plt.savefig(exp_name + ".png")

    return controller, avg_time, costs, best_cost


def agnostic_sys_id_tracking_learner(linearized_model: bool, pdl: bool):
    np.random.seed(0)
    model, index, env = setup_env()
    agnostic_sys_id_tracking_learner_(env, model, index, linearized_model, pdl)


if __name__ == "__main__":
    agnostic_sys_id_tracking_learner(True, False)
