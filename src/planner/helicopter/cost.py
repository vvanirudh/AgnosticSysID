import numpy as np

LARGE_VALUE = 1e10


def cost_state(x_result, x_target, Q):
    return min(
        0.5 * np.append(x_result - x_target, 1).T @ (Q @ (np.append(x_result - x_target, 1))),
        LARGE_VALUE,
    )


def cost_control(u_result, u_target, R):
    return min(0.5 * (u_result - u_target).T @ (R @ (u_result - u_target)), LARGE_VALUE)


def cost_final(x_result, x_target, Qfinal):
    return min(
        0.5 * np.append(x_result - x_target, 1).T @ (Qfinal @ (np.append(x_result - x_target, 1))),
        LARGE_VALUE,
    )


def cost_trajectory(x_result, u_result, x_target, u_target, Q, R, Qfinal):
    H = u_result.shape[1]
    cost = 0.0
    for t in range(H):
        cost += cost_state(x_result[:, t], x_target[:, t], Q)
        cost += cost_control(u_result[:, t], u_target[:, t], R)

    cost += cost_final(x_result[:, H], x_target[:, H], Qfinal)
    return cost
