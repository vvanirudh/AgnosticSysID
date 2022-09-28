import numpy as np


def linearized_heli_dynamics_2(
    xstar0, xstar1, ustar0, dt, helicopter_model, helicopter_index, helicopter_env
):
    """% Computes linearized dynamics of the form:
    % [x(t+dt) - xstar1; 1] = A [x(t) - xstar0; 1] + B (u(t) - ustar0)
    % xstar1 might be your target state at the next time
    % xstar0 might be your target state at the current time
    % ustar0 might be your trim control input at current time"""
    x1 = helicopter_env.step(xstar0, ustar0, dt, helicopter_model, helicopter_index)

    A = []
    B = []

    epsilon = np.ones(xstar0.shape[0]) * 0.01
    for i in range(epsilon.shape[0]):
        delta = np.zeros(xstar0.shape[0])
        delta[i] = epsilon[i]
        fx_t1m = helicopter_env.step(xstar0 - delta, ustar0, dt, helicopter_model, helicopter_index)
        fx_t1p = helicopter_env.step(xstar0 + delta, ustar0, dt, helicopter_model, helicopter_index)

        A.append((fx_t1p - fx_t1m) / epsilon[i] / 2)

    epsilon = np.ones(ustar0.shape[0]) * 0.01
    for i in range(epsilon.shape[0]):
        delta = np.zeros(ustar0.shape[0])
        delta[i] = epsilon[i]
        fx_t1m = helicopter_env.step(xstar0, ustar0 - delta, dt, helicopter_model, helicopter_index)
        fx_t1p = helicopter_env.step(xstar0, ustar0 + delta, dt, helicopter_model, helicopter_index)

        B.append((fx_t1p - fx_t1m) / epsilon[i] / 2)

    A.append(x1 - xstar1)

    A = np.array(A).T
    B = np.array(B).T

    last_row_A = np.zeros((1, A.shape[0] + 1))
    last_row_A[0, -1] = 1

    last_row_B = np.zeros((1, B.shape[1]))

    A = np.vstack([A, last_row_A])
    B = np.vstack([B, last_row_B])

    return A, B


def linearized_heli_dynamics(
    xt0, xt1, x0, ut0, u0, dt, helicopter_model, helicopter_index, helicopter_env
):
    """
        % Linearizes the dynamics around the state x0 and control position u0
    % The system is parametrized in coordinates relative to the position and
    % orientation of a state xt0 at the current time, and with respect to
    % xt1 at the next time step.
    % returns A, B, such that
    %       [x(t+dt)-xt1 ; 1] = A*[ x(t)-xt0; 1]  + B* [u(t)-ut0];
    %    is the linear approximation at x0,u0
    % Note: it adds an additional "1" entry to the state to avoid requiring an
    % offset term
    """
    x1 = helicopter_env.step(x0, u0, dt, helicopter_model, helicopter_index)

    A = []
    B = []

    epsilon = np.ones(x0.shape[0]) * 0.01
    for i in range(epsilon.shape[0]):
        delta = np.zeros(x0.shape[0])
        delta[i] = epsilon[i]
        fx_t1m = helicopter_env.step(x0 - delta, u0, dt, helicopter_model, helicopter_index)
        fx_t1p = helicopter_env.step(x0 + delta, u0, dt, helicopter_model, helicopter_index)

        A.append((fx_t1p - fx_t1m) / epsilon[i] / 2)

    epsilon = np.ones(u0.shape[0]) * 0.01
    for i in range(epsilon.shape[0]):
        delta = np.zeros(u0.shape[0])
        delta[i] = epsilon[i]
        fx_t1m = helicopter_env.step(x0, u0 - delta, dt, helicopter_model, helicopter_index)
        fx_t1p = helicopter_env.step(x0, u0 + delta, dt, helicopter_model, helicopter_index)

        B.append((fx_t1p - fx_t1m) / epsilon[i] / 2)

    A = np.array(A).T
    B = np.array(B).T

    last_column_A = (x1 - xt1 + A @ (xt0 - x0) + B @ (ut0 - u0)).reshape(-1, 1)
    A = np.hstack([A, last_column_A])

    last_row_A = np.zeros((1, A.shape[0] + 1))
    last_row_A[0, -1] = 1

    last_row_B = np.zeros((1, B.shape[1]))

    A = np.vstack([A, last_row_A])
    B = np.vstack([B, last_row_B])

    return A, B
