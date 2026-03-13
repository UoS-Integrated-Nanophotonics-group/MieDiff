import numpy as np
import torch

import pymiediff as pmd


def test_pena_core_shell_parity_with_scipy_backend():
    k0 = 2 * torch.pi / torch.linspace(450.0, 900.0, 11)
    kwargs = dict(
        k0=k0,
        r_c=70.0,
        eps_c=(2.2 + 0.2j) ** 2,
        r_s=110.0,
        eps_s=(1.6 + 0.05j) ** 2,
        eps_env=1.0,
        n_max=12,
    )

    ref = pmd.multishell.mie_coefficients(**kwargs, backend="scipy")
    out = pmd.multishell.mie_coefficients(**kwargs, backend="pena")

    torch.testing.assert_close(out["a_n"], ref["a_n"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out["b_n"], ref["b_n"], atol=1e-6, rtol=1e-6)


def test_pena_multilayer_shapes_and_metadata():
    k0 = 2 * torch.pi / torch.linspace(500.0, 800.0, 5)
    r_layers = torch.tensor([40.0, 75.0, 120.0], dtype=torch.float64)
    eps_layers = torch.tensor(
        [(1.8 + 0.1j) ** 2, (2.4 + 0.0j) ** 2, (1.5 + 0.05j) ** 2],
        dtype=torch.complex128,
    )

    out = pmd.multishell.mie_coefficients(
        k0=k0,
        r_layers=r_layers,
        eps_layers=eps_layers,
        eps_env=1.0,
        backend="pena",
        n_max=10,
    )

    assert out["a_n"].shape == (10, 1, 5)
    assert out["b_n"].shape == (10, 1, 5)
    assert int(out["L"].item()) == 3
    assert out["r_layers"].shape == (1, 3)
    assert out["eps_layers"].shape == (1, 3, 5)
    assert out["m_layers"].shape == (1, 3, 5)


def test_pena_return_internal_not_implemented():
    k0 = 2 * torch.pi / torch.tensor([600.0])

    with np.testing.assert_raises(NotImplementedError):
        pmd.multishell.mie_coefficients(
            k0=k0,
            r_c=60.0,
            eps_c=(2.0 + 0.0j) ** 2,
            r_s=100.0,
            eps_s=(1.4 + 0.1j) ** 2,
            eps_env=1.0,
            backend="pena",
            return_internal=True,
            n_max=8,
        )


def test_pena_auto_nmax_uses_pena2009_rule():
    k0 = 2 * torch.pi / torch.linspace(500.0, 1000.0, 7)
    r_layers = torch.tensor([40.0, 75.0, 120.0, 170.0], dtype=torch.float64)
    eps_layers = torch.tensor(
        [(2.2 + 0.1j) ** 2, (1.8 + 0.0j) ** 2, (1.5 + 0.04j) ** 2, (1.3 + 0.0j) ** 2],
        dtype=torch.complex128,
    )

    out = pmd.multishell.mie_coefficients(
        k0=k0,
        r_layers=r_layers,
        eps_layers=eps_layers,
        eps_env=1.33**2,
        backend="pena",
        n_max=None,
    )

    # inputs arrive broadcasted in the output dict; use these for exact comparison
    expected = pmd.helper.get_truncution_criteroin_pena2009(
        k0=out["k0"],
        r_layers=out["r_layers"],
        eps_layers=out["eps_layers"],
        eps_env=out["eps_env"],
    )
    assert int(out["n_max"].item()) == int(expected)


def test_pena_multilayer_angular_scattering():
    k0 = 2 * torch.pi / torch.linspace(500.0, 800.0, 4)
    theta = torch.linspace(0.1, 2.9, 16)
    r_layers = torch.tensor([45.0, 70.0, 105.0, 150.0], dtype=torch.float64)
    eps_layers = torch.tensor(
        [(2.0 + 0.1j) ** 2, (1.7 + 0.0j) ** 2, (1.4 + 0.04j) ** 2, (1.3 + 0.0j) ** 2],
        dtype=torch.complex128,
    )

    out = pmd.multishell.angular_scattering(
        k0=k0,
        theta=theta,
        r_layers=r_layers,
        eps_layers=eps_layers,
        eps_env=1.2**2,
        backend="pena",
        n_max=30,
    )

    assert out["S1"].shape == (1, 4, 16)
    assert out["S2"].shape == (1, 4, 16)
    assert out["i_unpol"].shape == (1, 4, 16)
    assert torch.isfinite(out["S1"].real).all()
    assert torch.isfinite(out["S1"].imag).all()


def test_particle_multilayer_with_real_materials_runs():
    wl0 = torch.linspace(600.0, 750.0, 4)
    k0 = 2 * torch.pi / wl0

    p = pmd.Particle(
        r_layers=torch.tensor([40.0, 75.0, 115.0, 160.0], dtype=torch.float64),
        mat_layers=[
            pmd.materials.MatDatabase("Si"),
            pmd.materials.MatDatabase("Ge"),
            pmd.materials.MatDatabase("Au"),
            pmd.materials.MatDatabase("Ag"),
        ],
        mat_env=1.33,
    )

    out = p.get_cross_sections(k0=k0)
    assert out["q_ext"].shape == (4,)
    assert torch.isfinite(out["q_ext"]).all()
    assert torch.isfinite(out["q_sca"]).all()
