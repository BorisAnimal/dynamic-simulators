import numpy as np


def harmonic_trajectory_builder(A0, A, freq):
    def trajectory(t):
        omega = 2 * np.pi * freq
        q_d = A0 + A * np.sin(omega * t)
        dq_d = A * omega * np.cos(omega * t)
        ddq_d = -A * omega ** 2 * np.sin(omega * t)
        return q_d, dq_d, ddq_d

    return trajectory


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    trajectory = harmonic_trajectory_builder(0, 1, 0.5 / np.pi)
    t = np.linspace(0, 2 * np.pi, 100)
    a, da, dda = trajectory(t)
    plt.plot(t, a)
    plt.show()
