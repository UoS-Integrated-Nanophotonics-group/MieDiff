import warnings

import numpy as np
import torch

import pymiediff as pmd


def test_multilayer_pena_vs_scattnlay_efficiencies():
    try:
        from scattnlay import scattnlay
    except ImportError:
        warnings.warn("`scattnlay` not installed. Skipping multilayer pena test.")
        return

    wl0 = 620.0  # nm
    k0 = 2 * np.pi / wl0

    n_env = 1.25
    r_layers = np.array([45.0, 80.0, 130.0], dtype=float)
    n_layers = np.array([2.0 + 0.2j, 1.55 + 0.0j, 1.35 + 0.08j], dtype=complex)
    m_layers = n_layers / n_env
    x_layers = k0 * n_env * r_layers

    terms, qext_ref, qsca_ref, *_ = scattnlay(x_layers, m_layers)

    out = pmd.coreshell.cross_sections(
        k0=torch.as_tensor([k0], dtype=torch.float64),
        r_layers=torch.as_tensor(r_layers, dtype=torch.float64),
        eps_layers=torch.as_tensor(n_layers**2, dtype=torch.complex128),
        eps_env=n_env**2,
        backend="pena",
        n_max=int(terms),
    )

    qext = out["q_ext"].squeeze().detach().cpu().numpy()
    qsca = out["q_sca"].squeeze().detach().cpu().numpy()

    np.testing.assert_allclose(qext, qext_ref, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(qsca, qsca_ref, atol=1e-6, rtol=1e-6)


def test_multilayer_pena_radiation_pattern_vs_scattnlay():
    try:
        from scattnlay import scattnlay
    except ImportError:
        warnings.warn("`scattnlay` not installed. Skipping multilayer pena test.")
        return

    wl0 = np.linspace(580.0, 740.0, 3)  # nm
    k0 = 2 * np.pi / wl0
    theta = np.linspace(0.05, np.pi - 0.05, 101)

    n_env = 1.22
    r_layers = np.array([45.0, 80.0, 130.0, 170.0], dtype=float)
    n_layers = np.array([2.0 + 0.1j, 1.65 + 0.0j, 1.4 + 0.06j, 1.3 + 0.0j], dtype=complex)

    out = pmd.coreshell.angular_scattering(
        k0=torch.as_tensor(k0, dtype=torch.float64),
        theta=torch.as_tensor(theta, dtype=torch.float64),
        r_layers=torch.as_tensor(r_layers, dtype=torch.float64),
        eps_layers=torch.as_tensor(n_layers**2, dtype=torch.complex128),
        eps_env=n_env**2,
        backend="pena",
        n_max=70,
    )
    i_per = out["i_per"][0].detach().cpu().numpy()
    i_par = out["i_par"][0].detach().cpu().numpy()
    i_unpol = out["i_unpol"][0].detach().cpu().numpy()

    i_per_ref = np.zeros_like(i_per)
    i_par_ref = np.zeros_like(i_par)
    i_unpol_ref = np.zeros_like(i_unpol)
    for i_wl, k0i in enumerate(k0):
        x_layers = k0i * n_env * r_layers
        m_layers = n_layers / n_env
        _, _, _, _, _, _, _, _, s1, s2 = scattnlay(x_layers, m_layers, theta=theta, nmax=70)
        i_per_ref[i_wl] = np.abs(s1) ** 2
        i_par_ref[i_wl] = np.abs(s2) ** 2
        i_unpol_ref[i_wl] = 0.5 * (i_per_ref[i_wl] + i_par_ref[i_wl])

    np.testing.assert_allclose(i_per, i_per_ref, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(i_par, i_par_ref, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(i_unpol, i_unpol_ref, atol=2e-5, rtol=2e-5)
