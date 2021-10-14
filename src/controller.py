import numpy as np

from impl.utils.lqr import lqr


class Controller:
    def __init__(self):
        pass

    def control(self, state, t):
        """
        :return: double
        """
        pass


class PDController(Controller):
    def __init__(self, trajectory, kp=2, kd=1):
        super().__init__()
        self.trajectory = trajectory
        self.kp = kp
        self.kd = kd

    def control(self, state, t):
        theta_d, dtheta_d, _ = self.trajectory(t)
        theta, dtheta = state

        return self.kp * (theta_d - theta) + self.kd * (dtheta_d - dtheta)


class ConstantController(Controller):
    def __init__(self, const=0.0):
        super().__init__()
        self.const = const

    def control(self, state, t):
        return self.const


class SquareWaveController(Controller):
    def __init__(self, A=0.0, per=1.0):
        super().__init__()
        self.A = A
        self.per = per

    def control(self, state, t):
        return self.A * (-1) ** round(t / self.per)


class LQR(Controller):
    def __init__(self, x0, A, B, u0, Q=None, R=None, Umax=None, lift=lambda x: x, discr=False):
        super().__init__()
        self.lift = lift
        self.Umax = Umax
        self.x0 = np.array(x0)
        self.x0_lifted = np.array(lift(x0))
        self.A = A
        self.B_inv = np.linalg.pinv(B)
        self.u0 = u0
        self.discr = discr

        if Q is None:
            Q = np.eye(A.shape[1])
        if R is None:
            R = np.eye(B.shape[1])
        _, K, _ = lqr(A, B, Q, R, discr=discr)
        self.K = K

    def control(self, state, t):
        """
        Срань захардкоженная. если будет баговать - переписать
        """
        state = np.array(state[:])
        state = self.lift(state.reshape(self.x0.shape))
        u = self.u0 + self.K.dot(state - self.x0_lifted).flatten()
        if self.Umax:
            return np.sign(u) * min(abs(u), self.Umax)
        else:
            return u
