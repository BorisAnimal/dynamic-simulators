import os
import pickle
import sys
from time import sleep

import numpy as np
from multiprocess import Process, Value, Array  # Yes, there could be red - just run code
from scipy.integrate import odeint

from src.plants.async_plant import AsyncPlant


class AsyncSimulator(AsyncPlant):

    def __init__(self, x0, data_file="last_run_default_save.pkl",
                 data_dir=os.path.join(os.path.dirname(__file__), "../../data"),
                 t=sys.maxsize, control_hz=100, between_steps=50, step_sleep_time=None):
        super().__init__()
        self.sim_time = t
        self.control_hz = control_hz
        self.between_steps = between_steps
        self.process = None
        self.state = Array('d', x0)
        self.t = Value('d', 0.0)
        self.u = Value('d', 0.0)
        self.xs_desc = []
        self.xs = []
        self.ts = []
        self.us = []
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, data_file)
        self.step_sleep_time = step_sleep_time

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
            ui = self.u.value
            xi = self.step(self.state[:], t, ui)
            # Save step results
            self.ts.append(t)
            self.xs_desc.append(xi[-1])
            self.xs.append(xi)
            self.us.append(ui)
            try:
                # from odeint
                self.state[:] = xi[-1]
            except:
                # from discrete plant
                self.state[:] = xi
            if self.step_sleep_time:
                sleep(self.step_sleep_time)

        self.on_success()

    def step(self, state, t, u):
        """
        Not depend on Multiprocessing. Could be used for data generation
        df :: (state_{i}, u_{i}) -> state_{i+1}
        """
        return odeint(
            self.dynamic, state,
            np.linspace(0, 1 / self.control_hz, self.between_steps),
            args=(u,)
        )

    def dynamic(self, state, t, u):
        """
        :return: derivative of state
        """
        pass

    def on_success(self):
        self.save_history()

    def save_history(self):
        obj = dict()
        for k, v in self.__dict__.items():
            try:
                pickle.dumps(v)
                obj[k] = v
            except:
                pass

        obj['xs_desc'] = np.vstack(self.xs_desc)
        obj['xs'] = np.vstack(self.xs)
        obj['ts'] = np.array(self.ts)
        obj['us'] = np.array(self.us)
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        pickle.dump(obj, open(self.data_file, "wb"))

    def load_history(self):
        obj = pickle.load(open(self.data_file, "rb"))
        return obj

    def terminate(self):
        self.process.terminate()

    def is_running(self):
        return self.process.is_alive()


if __name__ == '__main__':
    """
    See impl in pendulum_plant.py
    """
