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
