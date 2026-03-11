import numpy as np
import torch
from scipy.special import spherical_jn, spherical_yn

import pymiediff as pmd


def _riccati_psi(n, z_np):
    return z_np * spherical_jn(n, z_np)


def _riccati_zeta(n, z_np):
    return z_np * (spherical_jn(n, z_np) + 1j * spherical_yn(n, z_np))


def _riccati_logderiv_1(n, z_np):
    psi = _riccati_psi(n, z_np)
    dpsi = spherical_jn(n, z_np) + z_np * spherical_jn(n, z_np, derivative=True)
    return dpsi / psi


def _riccati_logderiv_3(n, z_np):
    zeta = _riccati_zeta(n, z_np)
    dzeta = (spherical_jn(n, z_np) + 1j * spherical_yn(n, z_np)) + z_np * (
        spherical_jn(n, z_np, derivative=True) + 1j * spherical_yn(n, z_np, derivative=True)
    )
    return dzeta / zeta


def test_pena_logderivatives_match_scipy():
    n_max = 8
    z = torch.linspace(0.5, 6.0, 12, dtype=torch.float64) + 1j * torch.linspace(
        0.1, 1.5, 12, dtype=torch.float64
    )

    D1 = pmd.special.pena_D1_n(n_max, z)
    D3 = pmd.special.pena_D3_n(n_max, z, D1=D1)
    z_np = z.numpy()

    ref_D1 = np.stack([_riccati_logderiv_1(n, z_np) for n in range(n_max + 1)], axis=0)
    ref_D3 = np.stack([_riccati_logderiv_3(n, z_np) for n in range(n_max + 1)], axis=0)

    np.testing.assert_allclose(D1.detach().numpy(), ref_D1, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(D3.detach().numpy(), ref_D3, atol=1e-8, rtol=1e-8)


def test_pena_Q_ratio_matches_definition():
    n_max = 6
    z1 = torch.linspace(0.4, 2.3, 9, dtype=torch.float64) + 1j * 0.3
    z2 = torch.linspace(0.8, 3.6, 9, dtype=torch.float64) + 1j * 0.7

    D1_z1 = pmd.special.pena_D1_n(n_max, z1)
    D3_z1 = pmd.special.pena_D3_n(n_max, z1, D1=D1_z1)
    D1_z2 = pmd.special.pena_D1_n(n_max, z2)
    D3_z2 = pmd.special.pena_D3_n(n_max, z2, D1=D1_z2)

    Q = pmd.special.pena_Q_n(
        n_max, z1=z1, z2=z2, D1_z1=D1_z1, D1_z2=D1_z2, D3_z1=D3_z1, D3_z2=D3_z2
    )

    z1_np = z1.numpy()
    z2_np = z2.numpy()
    ref_Q = []
    for n in range(n_max + 1):
        ref_Q.append(
            (_riccati_psi(n, z1_np) * _riccati_zeta(n, z2_np))
            / (_riccati_psi(n, z2_np) * _riccati_zeta(n, z1_np))
        )
    ref_Q = np.stack(ref_Q, axis=0)
    np.testing.assert_allclose(Q.detach().numpy(), ref_Q, atol=1e-8, rtol=1e-8)
