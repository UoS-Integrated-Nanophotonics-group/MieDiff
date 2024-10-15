# -*- coding: utf-8 -*-
"""
main routines of pymiediff
"""
# %%
import warnings

import numpy as np
import torch
from pymiediff import special  # use absolute package internal imports!


def sph_h1n(z, n):
    return special.Jn(z, n) + 1j*special.Yn(z, n)

def sph_h1n_der(z, n):
    return special.dJn(z, n) + 1j*special.dYn(z, n)

def psi(z, n):
    return z*special.Jn(z,n)

def chi(z, n):
    return -z*special.Yn(z, n)

def xi(z, n):
    return z*sph_h1n(z, n)

def psi_der(z, n):
    return special.Jn(z,n) + z*special.dJn(z,n)

def chi_der(z, n):
    return -special.Yn(z,n) - z*special.dYn(z,n)

def xi_der(z, n):
    return sph_h1n(z,n) + z*sph_h1n_der(z,n)


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

    from special import Jn

    N_pt_test = 100
    N_order_test = 1

    n = torch.tensor(5)
    # z = torch.rand(3, dtype=torch.complex64).unsqueeze(0)
    # Jn
    z = torch.linspace(1, 10, N_pt_test) + 1j * torch.linspace(0.5, 3, N_pt_test)
    z.requires_grad = True

    n1 = torch.tensor(1)
    n2 = torch.tensor(2)
    n3 = torch.tensor(3)

    test = Jn(z, n)
    test2 = psi(z, n2)
    test3 = psi(z, n3)

    # test = test1 + test2 + test3

    z_plot = z.detach().numpy().squeeze()
    a1_plot = test.detach().numpy().squeeze()

    plt.plot(z_plot, a1_plot.real)
    plt.show()
# define here functions / classes that should be provided by the `main` module of the package