import numpy as np
from scipy.integrate import odeint


# Самый простой симулятор без процессов и всего прочего
def simulator(control, dynamic, x0, system_params, t=2, control_hz=100, between_steps=40, **kwargs):
    total_steps = t * control_hz * between_steps
    control_params = kwargs.get('control_params', system_params)

    x = None
    for t in range(int(t * control_hz)):
        u = control(x0, **kwargs)
        xi = odeint(dynamic, x0, np.linspace(0, 1 / control_hz, between_steps), args=(u, system_params))
        if x is not None:
            x = np.vstack([x, xi])
        else:
            x = xi
        x0 = x[-1]
    return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    control = lambda state, *args: 0.2
    dynamic = lambda x, t, u, params: [x[1],
                                       (u - params['b'] * x[1] - params['m'] * params['g'] * params['l'] * np.sin(
                                           x[0])) / (params['m'] * params['l'] ** 2)]
    sys_params = {
        'm': 1,
        'g': 9.81,
        'l': 1,
        'b': 0.1,
    }

    x = simulator(control, dynamic, [0, 0], sys_params, t=50)
    plt.plot(x[:, 0], label='theta')
    plt.plot(x[:, 1], label='dtheta')
    plt.legend()
    plt.show()
