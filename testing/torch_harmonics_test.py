from torch_harmonics.legendre import legpoly
import numpy as np
import matplotlib.pyplot as plt
import torch
m = 0
n_theta = 80

def precompute_legpoly(mmax, lmax, t, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(\cos \theta) at the positions specified by t (theta).
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982;
        https://apps.dtic.mil/sti/citations/ADA123406
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients
    """

    return legpoly(mmax, lmax, torch.cos(t), norm=norm, inverse=inverse, csphase=csphase)



teq = torch.linspace(0, np.pi, n_theta, dtype=torch.float64, requires_grad= True)

mmax = torch.tensor(n_theta, requires_grad=False)
lmax = torch.tensor(n_theta, requires_grad=False)


def pct(m_max, l_max, teq):
    return torch.from_numpy(np.sqrt(2 * np.pi) * precompute_legpoly(m_max, l_max, teq, norm = "schmidt"))


result = pct(mmax, lmax, teq)

grad = torch.autograd.grad(
    outputs=result, inputs=[teq], grad_outputs=torch.ones_like(result)
)


print("Grad:", grad)

fig, ax = plt.subplots(1, 1)
for l in range(4):
    ax.plot(np.cos(teq), pct[0, l])
plt.show()