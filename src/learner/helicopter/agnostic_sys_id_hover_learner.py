from src.env.helicopter.helicopter_model import dt
from src.env.helicopter.helicopter_env import setup_env
from src.learner.helicopter.noise import get_hover_noise
from src.planner.helicopter.helicopter_hover import (
    hover_at_zero,
    hover_controller,
    hover_trims,
    test_hover_controller_,
)
from src.learner.helicopter.exploration_distribution import (
    desired_hover_trajectory_exploration_distribution,
    expert_hover_exploration_distribution,
)
from src.learner.helicopter.fit_model import (
    fit_linearized_model,
    fit_linearized_model_moment,
    fit_parameterized_model,
    initial_linearized_model,
    initial_parameterized_model,
)
from src.learner.helicopter.evaluate_controller import (
    optimal_hover_controller_for_linearized_model,
    evaluate_hover_controller,
    optimal_hover_controller_for_parameterized_model_psdp,
    optimal_hover_controller_for_parameterized_model_ilqr,
)

import numpy as np
import ray
from collections import deque
import matplotlib.pyplot as plt
import time
import argparse
import pickle as pkl


def get_optimal_hover_controller(model, H, linearized_model: bool, pdl: bool):
    if linearized_model:
        controller = optimal_hover_controller_for_linearized_model(model)
    elif pdl:
        controller = optimal_hover_controller_for_parameterized_model_psdp(model)
    else:
        controller = optimal_hover_controller_for_parameterized_model_ilqr(model, H)
    return controller


def get_initial_hover_model(H, linearized_model: bool):
    return (
        initial_linearized_model(H)
        if linearized_model
        else initial_parameterized_model(use_true_model=False)
    )


def agnostic_sys_id_hover_learner_(
    helicopter_env,
    helicopter_model,
    helicopter_index,
    linearized_model: bool,
    pdl: bool,
    num_iterations=100,
    num_samples_per_iteration=100,
    exploration_distribution_type="expert_controller_with_noise",
    plot=True,
    add_noise=True,
):
    H = 100
    nominal_model = get_initial_hover_model(H, linearized_model)
    model = get_initial_hover_model(H, linearized_model)
    controller = get_optimal_hover_controller(model, H, linearized_model, pdl)
    dataset = deque(maxlen=10000)

    if exploration_distribution_type == "desired_trajectory":
        exploration_distribution = desired_hover_trajectory_exploration_distribution(
            H, 0.0025, 0.0001
        )
    elif exploration_distribution_type == "expert_controller":
        exploration_distribution = expert_hover_exploration_distribution(
            helicopter_env, helicopter_model, helicopter_index, H, 0.0, 0.0, add_noise=add_noise
        )
    elif exploration_distribution_type == "expert_controller_with_noise":
        exploration_distribution = expert_hover_exploration_distribution(
            helicopter_env, helicopter_model, helicopter_index, H, 0.0, 0.0001, add_noise=add_noise
        )
    else:
        raise NotImplementedError("Unknown exploration distribution type")

    costs = []
    x_target = np.array([hover_at_zero for _ in range(H + 1)]).T
    u_target = np.array([hover_trims for _ in range(H)]).T
    # Evaluate controller
    costs.append(
        evaluate_hover_controller(
            controller,
            x_target,
            u_target,
            helicopter_model,
            helicopter_index,
            helicopter_env,
            H,
            add_noise=False,
        )
    )
    print(costs[-1])

    total_time = 0.0
    for n in range(num_iterations):
        print("Iteration", n)
        # Rollout controller in real world
        x_result, u_result, _ = test_hover_controller_(
            controller,
            helicopter_model,
            helicopter_index,
            helicopter_env,
            H,
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
                    noise=get_hover_noise() if add_noise else np.zeros(6),
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
            fit_linearized_model(
                dataset,
                nominal_model,
                nominal_state=hover_at_zero,
                nominal_control=hover_trims,
                nominal_next_state=hover_at_zero,
            )
            if linearized_model
            else fit_parameterized_model(dataset, nominal_model, previous_model=model)
        )

        # Compute new optimal controller
        start = time.time()
        controller = get_optimal_hover_controller(model, H, linearized_model, pdl)
        end = time.time()
        # Evaluate controller
        costs.append(
            evaluate_hover_controller(
                controller,
                x_target,
                u_target,
                helicopter_model,
                helicopter_index,
                helicopter_env,
                H,
                add_noise=False,
            )
        )
        print(costs[-1])

        total_time += end - start

    avg_time = total_time / num_iterations
    best_controller = optimal_hover_controller_for_parameterized_model_ilqr(helicopter_model, H)
    best_cost = evaluate_hover_controller(
        best_controller,
        x_target,
        u_target,
        helicopter_model,
        helicopter_index,
        helicopter_env,
        H,
        add_noise=False,
    )
    print("Cost of optimal trajectory is", best_cost)

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
        # plt.show()

    return controller, avg_time, costs, best_cost, model


def agnostic_sys_id_hover_learner(
    linearized_model: bool, pdl: bool, plot=True, add_noise=True, num_iterations=100
):
    np.random.seed(0)
    model, index, env = setup_env()
    return agnostic_sys_id_hover_learner_(
        env,
        model,
        index,
        linearized_model,
        pdl,
        plot=plot,
        add_noise=add_noise,
        num_iterations=num_iterations,
    )


def agnostic_sys_id_hover_experiment(add_noise=True, num_iterations=100):
    _, ag_time, ag_costs, best_cost, ag_model = agnostic_sys_id_hover_learner(
        False, False, plot=False, add_noise=add_noise, num_iterations=num_iterations
    )
    _, pdl_time, pdl_costs, _, pdl_model = agnostic_sys_id_hover_learner(
        False, True, plot=False, add_noise=add_noise, num_iterations=num_iterations
    )

    filename = "data/" + "agnostic_sys_id_" + str(num_iterations) + "_hover_" + str(add_noise)
    np.save(filename + ".npy", np.array(ag_costs))
    pkl.dump(ag_model, open(filename + ".pkl", "wb"))
    filename = "data/" + "pdl_" + str(num_iterations) + "_hover_" + str(add_noise)
    np.save(filename + ".npy", pdl_costs)
    pkl.dump(pdl_model, open(filename + ".pkl", "wb"))

    xrange = len(ag_costs)
    plt.plot(np.arange(xrange), ag_costs, label="Agnostic SysID " + str(ag_time))
    plt.plot(np.arange(xrange), pdl_costs, label="PDL " + str(pdl_time))
    plt.plot(
        np.arange(xrange),
        [best_cost for _ in range(xrange)],
        "--",
        label="Opt",
    )
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.yscale("log")
    plt.savefig("hover_exp.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-noise", action="store_true", default=False)
    parser.add_argument("--num_iterations", type=int, default=100)
    args = parser.parse_args()
    ray.init()
    agnostic_sys_id_hover_experiment(
        add_noise=(not args.no_noise), num_iterations=args.num_iterations
    )
