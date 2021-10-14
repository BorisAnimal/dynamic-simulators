from src.controller import *
from src.plants.async_simulator import AsyncSimulator
from src.runner import Runner


class TSALinDynamic:
    def __init__(self, **kwargs):
        # From sim https://colab.research.google.com/drive/11R1rL7A0xAfN7o9110E4vitrMVVuk73t#scrollTo=0bXNx27uPqRz
        L0 = 160 * 1e-3
        r0 = 0.74 * 1e-3
        m = 2.85
        I = 9.48e-6
        tau_c = 1.91 * 1e-3
        b_theta = 1.31 * 1e-6
        F_c = 4.11
        b_x = 9.46
        Kr_star = 11.75 * 1e3
        KL_star = 9.98 * 1e3

        g = 9.81

        self.L0 = kwargs.get("L0", L0)
        self.r0 = kwargs.get("r0", r0)
        self.m = kwargs.get("m", m)
        self.I = kwargs.get("I", I)
        self.g = kwargs.get("g", g)
        self.b_theta = kwargs.get("b_theta", b_theta)
        self.b_x = kwargs.get("b_x", b_x)

    def __call__(self, state, t, u):
        u = 0.001 * u
        [theta, dtheta] = state
        L0, r0, m, I, g, b_theta, b_x = self.L0, self.r0, self.m, self.I, self.g, self.b_theta, self.b_x
        # Jacobian
        L_X = np.sqrt(L0 ** 2 - (theta * r0) ** 2)
        J = theta * r0 ** 2 / (L_X)
        dx = J * dtheta
        x = L0 - L_X
        tmp = L0 ** 2 - r0 ** 2 * theta ** 2
        dJ = dtheta * r0 ** 2 * (1 / np.sqrt(tmp) + r0 ** 2 * theta ** 2 / tmp ** 1.5)

        # ddtheta = 1/(I + m*J**2) * (u - J*m*g - J*m*dJ * dtheta)
        ddtheta = (u - J * (m * g + b_x * dx) - m * J * dJ * dtheta - b_theta * dtheta) / (I + m * J ** 2)
        # ddtheta = 1/(I+J**2*m*R/r)*(u-J*(m*R/r*(dJ*dtheta+g) + b_alpha*dalpha) - b_theta*dtheta) # Rot
        return np.array([dtheta, ddtheta, ])


class TSALinAsyncPlant(AsyncSimulator):
    def __init__(self, x0, t, data_file='tsa_results.pkl', **kwargs):
        super().__init__(x0=x0, t=t, data_file=data_file, **kwargs)
        # specified params
        self._dynamic = TSALinDynamic(**kwargs)

    def dynamic(self, state, t, u):
        return self._dynamic(state, t, u)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ## Setup environment
    plant = TSALinAsyncPlant(x0=[0.0, 100.0], t=3)
    # plant = CartpoleAsyncPlant(t0=[-0.1, -0.5], t=3)
    # controller = ConstantController(0.0)
    controller = SquareWaveController(1, 0.3)
    # Run simulation
    Runner(plant, controller).run()
    # Load history and prepare for plotting
    fields = plant.load_history()
    data = fields['xs_desc']
    ts = fields['ts']
    print("Result data shape:", data.shape)
    x1, x2 = data[:, 0], data[:, 1]
    u = fields['us']
    # Plot data
    t = lambda x: np.linspace(0, plant.t.value, len(x))
    plt.plot(t(x1), x1, label='x1')
    plt.plot(t(x2), x2, label='x2')
    # plt.plot(t(u), u)
    plt.legend()
    plt.show()
