from scipy.linalg import solve_discrete_are
from numpy.linalg import solve
import numpy as np


def lqr_lti(A, B, Q, R):
    # P = solve_discrete_are(A, B, Q, R)
    P = np.ones_like(A)
    K = solve(R + B.T.dot(P.dot(B)), -B.T.dot(P.dot(A)))
    while True:
        P = Q + K.T.dot(R.dot(K)) + (A + B.dot(K)).T.dot(P.dot(A + B.dot(K)))
        Knew = solve(R + B.T.dot(P.dot(B)), -B.T.dot(P.dot(A)))
        if np.linalg.norm(Knew - K) < 1e-6:
            break
        K = Knew.copy()
    return K, P


def lqr_ltv(A, B, Q, R, Qfinal):
    H = len(A)
    P = [np.zeros_like(A[0]) for k in range(H + 1)]
    K = [np.zeros_like(B[0]).T for k in range(H)]

    P[H] = Qfinal.copy()
    for t in range(H - 1, -1, -1):
        K[t] = solve(R + B[t].T.dot(P[t + 1].dot(B[t])), -B[t].T.dot(P[t + 1].dot(A[t])))
        P[t] = (
            Q
            + K[t].T.dot(R.dot(K[t]))
            + (A[t] + B[t].dot(K[t])).T.dot(P[t + 1].dot(A[t] + B[t].dot(K[t])))
        )

    return K, P


def lqr_linearized_tv(A, B, C_x, C_u, C_xx, C_uu):
    H = len(A)

    k = [np.zeros(B[0].shape[1]) for _ in range(H)]
    K = [np.zeros_like(B[0]).T for _ in range(H)]
    V_x = np.zeros(A[0].shape[0])
    V_xx = np.zeros_like(A[0])

    for t in range(H - 1, -1, -1):
        A_t, B_t = A[t], B[t]
        C_x_t, C_u_t, C_xx_t, C_uu_t = C_x[t], C_u[t], C_xx[t], C_uu[t]

        Q_x = C_x_t + A_t.T.dot(V_x)
        Q_u = C_u_t + B_t.T.dot(V_x)

        Q_xx = C_xx_t + A_t.T.dot(V_xx.dot(A_t))
        Q_ux = B_t.T.dot(V_xx.dot(A_t))
        Q_uu = C_uu_t + B_t.T.dot(V_xx.dot(B_t))

        K.append(solve(-Q_uu, Q_ux))
        k.append(solve(-Q_uu, Q_u))

        V_x = Q_x - K[t].T.dot(Q_uu.dot(k[t]))
        V_xx = Q_xx - K[t].T.dot(Q_uu.dot(K[t]))
        V_xx = (V_xx + V_xx.T) / 2.0

    return k, K
