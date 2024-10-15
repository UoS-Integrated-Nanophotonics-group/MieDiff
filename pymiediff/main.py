# -*- coding: utf-8 -*-
"""
main routines of pymiediff
"""
# %%
import warnings

import numpy as np
import torch
#from . import special
from pymiediff import special  # use absolute package internal imports!


def sph_h1n(z, n):
    return special.Jn(n, z) + 1j*special.Yn(n, z)


def sph_h1n_der(z, n):
    return special.dJn(n, z) + 1j*special.dYn(n, z)


def psi(z, n):
    return z*special.Jn(n, z)


def chi(z, n):
    return -z*special.Yn(n, z)


def xi(z, n):
    return z*sph_h1n(n, z)


def psi_der(z, n):
    return special.Jn(n, z) + z*special.dJn(n, z)


def chi_der(z, n):
    return -special.Yn(n, z) - z*special.dYn(n, z)


def xi_der(z, n):
    return sph_h1n(z, n) + z*sph_h1n_der(z, n)


def An(x, n, m1, m2):
    return (m2*psi(m2*x, n)*psi_der(m1*x, n) - m1*psi_der(m2*x, n)*psi(m1*x, n))/(m2*chi(m2*x, n)*psi_der(m1*x, n) - m1*chi_der(m2*x, n)*psi(m1*x, n))


def Bn(x, n, m1, m2):
    return (m2*psi(m1*x, n)*psi_der(m2*x, n) - m1*psi(m2*x, n)*psi_der(m1*x, n))/(m2*chi_der(m2*x, n)*psi(m1*x, n) - m1*psi_der(m1*x, n)*chi(m2*x, n))


def an(x, y, n, m1, m2):
    return (psi(y, n)*(psi_der(m2*y, n) - An(x, n, m1, m2)*chi_der(m2*y, n)) - m2*psi_der(y, n)*(psi(m2*y, n) - An(x, n, m1, m2)*chi(m2*y, n)))/(xi(y, n)*(psi_der(m2*y, n) - An(x, n, m1, m2)*chi_der(m2*y, n)) - m2*xi_der(y, n)*(psi(m2*y, n) - An(x, n, m1, m2)*chi(m2*y, n)))


def bn(x, y, n, m1, m2):
    return (m2*psi(y, n)*(psi_der(m2*y, n) - Bn(x, n, m1, m2)*chi_der(m2*y, n)) - psi_der(y, n)*(psi(m2*y, n) - Bn(x, n, m1, m2)*chi(m2*y, n)))/(m2*xi(y, n)*(psi_der(m2*y, n) - Bn(x, n, m1, m2)*chi_der(m2*y, n)) - xi_der(y, n)*(psi(m2*y, n) - Bn(x, n, m1, m2)*chi(m2*y, n)))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # from special import Jn

    N_pt_test = 200
    N_order_test = 1

    n = torch.tensor(5)

    m1 =  3.0 # n_core / n_env
    m2 =  2.0 # n_shell / n_env

    x = torch.linspace(1, 20, N_pt_test) + 1j * torch.linspace(0.5, 3, N_pt_test)
    y = torch.linspace(1, 20, N_pt_test) + 1j * torch.linspace(0.5, 3, N_pt_test)

    m1 = torch.tensor(m1, requires_grad=True)
    m2 = torch.tensor(m2, requires_grad=True)

    print(x)

    #a1 = psi(n, x)
    a1 = an(x, y, n, m1, m2)

    x_plot = x.detach().numpy().squeeze()
    a1_plot = a1.detach().numpy().squeeze()

    plt.plot(x_plot, a1_plot)
    plt.show()
# define here functions / classes that should be provided by the `main` module of the package