import numpy as np


class Pendubot:

    def __init__(self, params):
        self.params = params

        def T(state):
            print(self)
            print(state)
            return params[0]

        self.T = T

    def D(self, state):
        q1, dq1, q2, dq2 = state
        m1, m2, lc1, l1, lc2, l2, I1, I2, g, b1, c1, b2, c2 = self.params

        d11 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * lc1 * lc2 * np.cos(q2)) + I1 + I2
        d12 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(q2)) + I2  # = d21
        d22 = m2 * lc2 ** 2 + I2

        return np.array([
            [d11, d12],
            [d12, d22],
        ])

    def C(self, state):
        q1, dq1, q2, dq2 = state
        m1, m2, lc1, l1, lc2, l2, I1, I2, g, b1, c1, b2, c2 = self.params

        h = -m2 * l1 * lc2 * np.sin(q2)
        return np.array([
            [h * dq2, h * dq2 + h * dq1],
            [-h * dq1, 0.0],
        ])

    def g(self, state):
        q1, dq1, q2, dq2 = state
        m1, m2, lc1, l1, lc2, l2, I1, I2, g, b1, c1, b2, c2 = self.params

        phi1 = (m1 * lc1 + m2 * l1) * g * np.cos(q1) + m2 * lc2 * g * np.cos(q1 + q2)
        phi2 = m2 * g * lc2 * np.cos(q1 + q2)

        return np.array([
            [phi1],
            [phi2],
        ])

    def ddq(self, state, torq=np.array([[0], [0]])):
        q1, dq1, q2, dq2 = state
        m1, m2, lc1, l1, lc2, l2, I1, I2, g, b1, c1, b2, c2 = self.params

        D_inv = np.linalg.inv(self.D(state))
        C = self.C(state)
        g = self.g(state)
        fric = - np.array([
            [c1 * np.tanh(dq1) + b1 * dq1],
            [c2 * np.tanh(dq2) + b2 * dq2],
        ])

        ddq = D_inv @ (- C @ np.array([[dq1], [dq2]]) - g + torq + fric)
        return ddq

    def K(self, state):
        q1, dq1, q2, dq2 = state
        dq = np.array([
            [dq1],
            [dq2],
        ])

        return 0.5 * dq.T @ self.D(state) @ dq

    def P(self, state):
        q1, dq1, q2, dq2 = state
        m1, m2, lc1, l1, lc2, l2, I1, I2, g, b1, c1, b2, c2 = self.params

        return (m1 * lc1 + m2 * l1) * g * np.sin(q1) + m2 * lc2 * g * np.sin(q1 + q2)
