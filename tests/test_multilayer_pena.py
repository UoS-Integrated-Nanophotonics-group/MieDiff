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
