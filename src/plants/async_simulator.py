import pickle
import sys
from multiprocessing import Process, Value, Array

import numpy as np
from scipy.integrate import odeint

from src.plants.async_plant import AsyncPlant
from src.trajectory import harmonic_trajectory_builder


class AsyncSimulator(AsyncPlant):

    def __init__(self, x0, t=sys.maxsize, control_hz=100, between_steps=50):
        self.sim_time = t
        self.control_hz = control_hz
        self.between_steps = between_steps
        self.process = None
        self.state = Array('d', x0)
        self.t = Value('d', 0.0)
        self.u = Value('d', 0.0)
        self.xs = []
        self.ts = []

        # specified params
        self.g = 9.81
        self.l = 1.0
        self.b = 0.1
        self.m = 1.0
        self.data_file = '../../data/data.csv'

    def get_state(self):
        return self.state, self.t.value

    def set_control(self, u):
        self.u.value = u

    def start(self, **kwargs):
        total_steps = self.sim_time * self.control_hz * self.between_steps
        control_params = kwargs.get('control_params', {})

        self.process = Process(target=self._execute)  # TODO: args
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
        [theta, dtheta] = state
        ddtheta = (u - self.b * dtheta - self.m * self.g * self.l * np.sin(
            theta)) / (self.m * self.l ** 2)

        return dtheta, ddtheta

    def terminate(self):
        self.process.terminate()

    def is_running(self):
        return self.process.is_alive()


class Controller:
    def __init__(self):
        self.trajectory = harmonic_trajectory_builder(0, np.pi / 6, 1)
        self.kp = 2
        self.kd = 1

    def control(self, state, t):
        theta_d, dtheta_d, _ = self.trajectory(t)
        theta, dtheta = state

        return self.kp * (theta_d - theta) + self.kd * (dtheta_d - dtheta)


class Runner:
    def __init__(self, plant: AsyncPlant, controller: Controller):
        self.plant = plant
        self.controller = controller

    def run(self):
        self.plant.start()
        while self.plant.is_running():
            state, t = self.plant.get_state()
            u = self.controller.control(state, t)
            self.plant.set_control(u)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sys_params = {
        'm': 1,
        'g': 9.81,
        'l': 1,
        'b': 0.1,
    }

    plant = AsyncSimulator([0, 0], t=5)
    plant.control_hz = 1000
    plant.between_steps = 20
    controller = Controller()
    controller.kp = 50
    controller.kd = 40

    Runner(plant, controller).run()
    fields = plant.load_history()
    data = fields['xs']
    ts = fields['ts']
    print(data.shape)
    x, dx = data[:, 0], data[:, 1]
    plt.plot(np.linspace(0, 1, len(x)), x, label='theta')
    plt.plot(np.linspace(0, 1, len(ts)), controller.trajectory(ts)[0], label='theta_d')
    plt.legend()
    plt.show()
