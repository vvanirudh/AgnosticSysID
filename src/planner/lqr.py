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
    return K


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

    return K
