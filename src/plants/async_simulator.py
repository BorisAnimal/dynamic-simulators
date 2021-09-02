import pickle
import sys
from multiprocessing import Process, Value, Array

import numpy as np
from scipy.integrate import odeint

from src.plants.async_plant import AsyncPlant
from src.controller import PDController
from src.runner import Runner
from src.trajectory import harmonic_trajectory_builder


class AsyncSimulator(AsyncPlant):

    def __init__(self, x0, data_file, t=sys.maxsize, control_hz=100, between_steps=50):
        super().__init__()
        self.sim_time = t
        self.control_hz = control_hz
        self.between_steps = between_steps
        self.process = None
        self.state = Array('d', x0)
        self.t = Value('d', 0.0)
        self.u = Value('d', 0.0)
        self.xs = []
        self.ts = []
        self.data_file = data_file

    def get_state(self):
        return self.state, self.t.value

    def set_control(self, u):
        self.u.value = u

    def start(self, **kwargs):
        total_steps = self.sim_time * self.control_hz * self.between_steps
        control_params = kwargs.get('control_params', {})

        self.process = Process(target=self._execute)
        self.process.start()

    def _execute(self):
        for t in np.linspace(0.0, self.sim_time, int(self.sim_time * self.control_hz)):
            self.t.value = t
            xi = odeint(
                self.dynamic, self.state[:],
                np.linspace(0, 1 / self.control_hz, self.between_steps),
                args=(self.u.value,)
            )

            self.ts.append(t)
            self.xs.append(xi)
            self.state[:] = xi[-1]
            # sleep(1 / self.control_hz)
        self.on_success()

    def on_success(self):
        self.save_history()

    def save_history(self):
        obj = self.__dict__
        del obj['state']
        del obj['u']
        del obj['t']
        del obj['process']
        obj['xs'] = np.vstack(self.xs)
        obj['ts'] = np.array(self.ts)
        pickle.dump(obj, open(self.data_file, "wb"))

    def load_history(self):
        obj = pickle.load(open(self.data_file, "rb"))
        return obj

    def dynamic(self, state, t, u):
        """
        :return: derivative of state
        """
        pass

    def terminate(self):
        self.process.terminate()

    def is_running(self):
        return self.process.is_alive()


class PendulumAsyncPlant(AsyncSimulator):
    def __init__(self, x0, g=9.81, l=1.0, b=0.1, m=1.0,
                 data_file='../../data/pendulum_data.csv',
                 t=sys.maxsize, control_hz=100, between_steps=50):
        super().__init__(x0=x0, data_file=data_file, t=t, control_hz=control_hz, between_steps=between_steps)
        # specified params
        self.g = g
        self.l = l
        self.b = b
        self.m = m

    def dynamic(self, state, t, u):
        [theta, dtheta] = state
        ddtheta = (u - self.b * dtheta - self.m * self.g * self.l * np.sin(
            theta)) / (self.m * self.l ** 2)

        return dtheta, ddtheta


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Setup environment
    plant = PendulumAsyncPlant(x0=[0, 0], t=5)
    trajectory = harmonic_trajectory_builder(A0=0, A=np.pi / 6, freq=1)
    controller = PDController(trajectory=trajectory, kp=50, kd=40)
    # Run simulation
    Runner(plant, controller).run()
    # Load history and prepare for plotting
    fields = plant.load_history()
    data = fields['xs']
    ts = fields['ts']
    print(data.shape)
    x, dx = data[:, 0], data[:, 1]
    # Plot data
    plt.plot(np.linspace(0, 1, len(x)), x, label='theta')
    plt.plot(np.linspace(0, 1, len(ts)), controller.trajectory(ts)[0], label='theta_d')
    plt.legend()
    plt.show()
