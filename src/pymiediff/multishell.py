"""
Multilayered scattering coefficients

analytical solutions taken from

REF
"""
import warnings

import torch

from pymiediff import special
from pymiediff import helper

def _Han(n, x, m, D1nz1, D1nz2, D1m1x1, D3nz1, l_max):

    # does this need to be an array? we only need the final l
    Hans = []

    for l_iter in range(1, l_max + 1):
        if l_iter == 1:
            Hans.append(special.D1n_torch())
        else:

            G1 = m[l_iter-1]*Hans[l_iter-2] - m[l_iter-2] * D1nz1[:, l_iter-1,...]
            G2 = m[l_iter-1]*Hans[l_iter-2] - m[l_iter-2] * D3nz1[:, l_iter-1,...]

            Qln = special.Ql_torch(n, l_iter, m, x)


            Hans.append(
                (G2 * D1nz2[:, l_iter-1, ...] - Qln * G1 * D3nz1[:, l_iter-1, ...]) \
                    / (G2 - Qln * G1)
            )

    Hans = torch.stack(Hans, dim=1)  # second dim: l layer

    return Hans


def _Hbn():

    pass


def _miecoef(
    x,
    n,
    m,
    return_internal=False,
    precision="double"
    ):

    l_max = 10

    # permeabilities set to 1
    mu1 = mu2 = mu = 1.0
    # get z1 = x_l m_l-1  and z2 = x_l m_l-1
    z1 = x[..., 1:] *  m[..., 0:-1]
    z2 = x[..., 1:] *  m[..., 1:]

    # Calculate logrithmic derivatives for all ns

    # D_n^(1) (x_l m_l)
    D1nz2 = special.D1n_torch(n, z2) # does not include x_1 m_1

    # D_n^(1) (x_1 m_1)
    D1m1x1 = special.D1n_torch(n, z2[..., 0])

    # D_n^(1) (x_l m_l-1)
    D1nz1 = special.D1n_torch(n, z1) # does not include x_1 m_0

    D3nz1 = special.D3n_torch(n, z1, D1nz1)


    # recurrence through all l
    Han = _Han(n, x, m, D1nz1, D1nz2, D1m1x1, D3nz1, l_max)

    pass


def mie_coefficients():
    pass


def _broadcast_mie_config_multilayer(k0, r, eps, eps_env):
    """
    Broadcast configs to 3 dimensions for multi-layer vectorization.

    Target dimension convention is (N particles, N layers, N wavevectors).

    Args:
        k0 (tensor of float): wavevector, shape (N_k0,)
        r (tensor of float): layer radii, shape (N_particles, N_layers)
        eps (tensor of complex): layer permittivity, shape (N_particles, N_layers)
                                 OR (N_particles, N_layers, N_k0) if dispersive
        eps_env (tensor of complex/float): environment permittivity, shape (N_k0,)

    Returns:
        k0, r, eps, eps_env broadcasted to compatible 3D shapes.
    """
    # 1. Process wavevectors (k0)
    k0 = torch.as_tensor(k0).squeeze()
    k0 = torch.atleast_1d(k0)
    assert k0.dim() == 1, "k0 must be a 1D array of wavevectors"
    N_k0 = k0.shape[0]

    # Add particle and layer dimensions -> Shape: (1, 1, N_k0)
    k0 = k0.unsqueeze(0).unsqueeze(0)

    # 2. Process radii (r)
    r = torch.as_tensor(r)
    assert r.dim() == 2, "r must be 2D with shape (N_particles, N_layers)"
    N_particles, N_layers = r.shape

    # Add wavevector dimension and stretch -> Shape: (N_particles, N_layers, N_k0)
    r = r.unsqueeze(-1).expand(-1, -1, N_k0)

    # 3. Process particle permittivities (eps)
    eps = torch.as_tensor(eps)

    if eps.dim() == 2:
        # If eps is constant across wavelengths (dispersionless)
        assert eps.shape == (N_particles, N_layers), "eps shape must match r"
        # Add wavevector dimension and stretch -> Shape: (N_particles, N_layers, N_k0)
        eps = eps.unsqueeze(-1).expand(-1, -1, N_k0)

    elif eps.dim() == 3:
        # If eps varies by wavelength (dispersive materials)
        assert eps.shape == (N_particles, N_layers, N_k0), "3D eps must match (N_particles, N_layers, N_k0)"
        # It's already the perfect shape, no broadcasting needed!

    else:
        raise ValueError("eps must be either 2D (dispersionless) or 3D (dispersive)")

    # 4. Process environment permittivity (eps_env)
    eps_env = torch.as_tensor(eps_env).squeeze()
    eps_env = torch.atleast_1d(eps_env)
    assert eps_env.dim() == 1 and eps_env.shape[0] == N_k0, "eps_env must be 1D and match N_k0 length"

    # Add particle and layer dimensions -> Shape: (1, 1, N_k0)
    eps_env = eps_env.unsqueeze(0).unsqueeze(0)

    return k0, r, eps, eps_env