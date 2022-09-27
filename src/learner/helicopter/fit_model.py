import numpy as np
import ray
from scipy.optimize import minimize
from src.env.helicopter.helicopter_env import (
    HelicopterEnv,
    step_batch_parameterized_model,
    step_parameterized_model,
)
from src.planner.helicopter.helicopter_hover import hover_at_zero, hover_trims
from src.env.helicopter.helicopter_model import (
    HelicopterIndex,
    LinearizedHelicopterModel,
    ParameterizedHelicopterModel,
    dt,
    HelicopterModel,
)

USE_RAY = False
if USE_RAY and not ray.is_initialized():
    ray.init()


def initial_linearized_model(H, time_varying=False):
    A = np.eye(13) if not time_varying else [np.eye(13) for _ in range(H)]
    B = np.eye(13, 4) if not time_varying else [np.eye(13, 4) for _ in range(H)]

    return LinearizedHelicopterModel(A, B, time_varying=time_varying)


def initial_parameterized_model():
    m = np.random.rand() * 10
    Ixx = np.random.rand()
    Iyy = np.random.rand()
    Izz = np.random.rand()

    Tx = np.array([0.1, -3, 14]) * Ixx
    # Tx = np.array([np.random.rand(), -6 * np.random.rand(), 28 * np.random.rand()]) * Ixx
    Ty = np.array([-0.1, -4, -9]) * Iyy
    # Ty = np.array([-np.random.rand(), -8 * np.random.rand(), -18 * np.random.rand()]) * Iyy
    Tz = np.array([0.1, -3, 14]) * Izz
    # Tz = np.array([np.random.rand(), -6 * np.random.rand(), 28 * np.random.rand()]) * Izz

    Fx = -0.1 * m
    Fy = np.array([0, -1]) * m
    Fz = np.array([-10, -1, -25]) * m
    # m = 5  # kg
    # Ixx = 0.3
    # Iyy = 0.3
    # Izz = 0.3

    # ## Aerodynamic forces parameters
    # Tx = np.array([0, -3.47, 13.20]) * Ixx
    # Ty = np.array([0, -3.06, -9.21]) * Iyy
    # Tz = np.array([0, -2.58, 14.84]) * Izz
    # Fx = -0.048 * m
    # Fy = np.array([0, -0.12]) * m
    # Fz = np.array([-9.81, -0.0005, -27.5]) * m

    return ParameterizedHelicopterModel(m, Ixx, Iyy, Izz, Tx, Ty, Tz, Fx, Fy, Fz)
    # return HelicopterModel()


def construct_training_data(dataset, about_hover_state: bool):
    states, controls, next_states = tuple(zip(*dataset))
    states, controls, next_states = np.array(states), np.array(controls), np.array(next_states)
    if about_hover_state:
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
    return states, controls, next_states


def fit_linearized_model(dataset, nominal_model):
    states, controls, next_states = construct_training_data(dataset, about_hover_state=True)

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


def fit_parameterized_model(dataset, nominal_model):
    states, controls, next_states = construct_training_data(dataset, about_hover_state=False)
    helicopter_env = HelicopterEnv()
    helicopter_index = HelicopterIndex()
    # TODO: Maybe we can learn residual on the nominal model?
    def loss_fn_(params):
        if not USE_RAY:
            model = ParameterizedHelicopterModel(
                *ParameterizedHelicopterModel.unpack_params(params)
            )
            predicted_next_states = np.array(
                [
                    helicopter_env.step(states[i], controls[i], dt, model, helicopter_index)
                    for i in range(states.shape[0])
                ]
            )
        else:
            predicted_next_states = step_batch_parameterized_model(states, controls, dt, params)
        return np.mean(np.linalg.norm(predicted_next_states - next_states, axis=1))

    result = minimize(
        loss_fn_,
        nominal_model.params,
        tol=0.01,
        options={"disp": True},
    )
    if result.success:
        return ParameterizedHelicopterModel(*ParameterizedHelicopterModel.unpack_params(result.x))
    else:
        raise Exception(result.message)


"""
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
"""
