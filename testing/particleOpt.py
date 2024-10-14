"""
test of autograd optimiser for particle
"""
import warnings

import torch
import numpy as np
import torch.nn.functional as F

from ..pymiediff.main import an, bn, An, Bn






if __name__ == "__main__":
    import matplotlib.pyplot as plt
    it = []
    losses = []

    n = torch.tensor(1)
    wlRes = 10
    wl = np.linspace(500,550, wlRes)
    r_core = 80.0
    r_shell = r_core + 100.0

    n_env = 1
    n_core = 4
    n_shell = 0.1   + .7j

    mu_env = 1
    mu_core = 1
    mu_shell = 1

    dtype = torch.complex64
    device = torch.device("cpu")

    n_max = 5
    k = 2 * np.pi / (wl / n_env)

    m1 = n_core / n_env
    m2 = n_shell / n_env
    x = k * r_core
    y = k * r_shell

    x = torch.tensor(x, requires_grad=True, dtype=dtype)
    y = torch.tensor(y, requires_grad=True, dtype=dtype)
    m1 = torch.tensor(m1, requires_grad=True, dtype=dtype)
    m2 = torch.tensor(m2, requires_grad=True, dtype=dtype)

    a1 = an(x, y, n, m1, m2)

    x_plot = x.detach().numpy().squeeze()
    a1_plot = a1.detach().numpy().squeeze()

    plt.plot(x_plot, a1_plot)
    plt.show()







