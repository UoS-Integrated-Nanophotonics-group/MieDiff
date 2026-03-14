import torch
import pytest

import pymiediff as pmd


def _make_configs():
    # Base two-layer particle
    r_layers_base = torch.tensor([80.0, 120.0], dtype=torch.double)
    eps_layers_base = torch.tensor(
        [(2.2 + 0.1j) ** 2, (1.5 + 0.05j) ** 2], dtype=torch.complex128
    )

    # Split inner layer into two identical materials
    r_layers_split = torch.tensor([40.0, 80.0, 120.0], dtype=torch.double)
    eps_layers_split = torch.tensor(
        [(2.2 + 0.1j) ** 2, (2.2 + 0.1j) ** 2, (1.5 + 0.05j) ** 2],
        dtype=torch.complex128,
    )

    return r_layers_base, eps_layers_base, r_layers_split, eps_layers_split


def test_multilayer_split_equivalence_spectra_and_nearfields():
    r_layers_base, eps_layers_base, r_layers_split, eps_layers_split = _make_configs()

    wl = torch.linspace(500.0, 800.0, 4, dtype=torch.double)
    k0 = 2 * torch.pi / wl
    eps_env = 1.0

    base_cs = pmd.multishell.cross_sections(
        k0=k0,
        r_layers=r_layers_base,
        eps_layers=eps_layers_base,
        eps_env=eps_env,
        backend="pena",
    )
    split_cs = pmd.multishell.cross_sections(
        k0=k0,
        r_layers=r_layers_split,
        eps_layers=eps_layers_split,
        eps_env=eps_env,
        backend="pena",
    )

    torch.testing.assert_close(base_cs["q_ext"], split_cs["q_ext"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(base_cs["q_sca"], split_cs["q_sca"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(base_cs["q_abs"], split_cs["q_abs"], rtol=1e-6, atol=1e-7)

    r_probe = torch.tensor(
        [
            [0.0, 0.0, 20.0],
            [0.0, 0.0, 60.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 150.0],
        ],
        dtype=torch.double,
    )

    scattnlay = pytest.importorskip("scattnlay")

    base_nf = pmd.multishell.nearfields(
        k0=k0,
        r_probe=r_probe,
        r_layers=r_layers_base,
        eps_layers=eps_layers_base,
        eps_env=eps_env,
        backend="pena",
    )
    split_nf = pmd.multishell.nearfields(
        k0=k0,
        r_probe=r_probe,
        r_layers=r_layers_split,
        eps_layers=eps_layers_split,
        eps_env=eps_env,
        backend="pena",
    )

    torch.testing.assert_close(base_nf["E_t"], split_nf["E_t"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(base_nf["H_t"], split_nf["H_t"], rtol=1e-6, atol=1e-7)

    # scattnlay comparison (reference for both cases)
    n_env = torch.sqrt(torch.as_tensor(eps_env, dtype=torch.complex128))
    n_env_np = complex(n_env.item())
    k = (k0 * n_env).detach().cpu().numpy()
    r_probe_np = r_probe.detach().cpu().numpy()

    def _scattnlay_fields(r_layers, eps_layers):
        mie = pmd.multishell.mie_coefficients(
            k0=k0,
            r_layers=r_layers,
            eps_layers=eps_layers,
            eps_env=eps_env,
            backend="pena",
        )
        n_max_use = int(mie["n_max"].item())

        r_l = r_layers.detach().cpu().numpy().astype(float)
        n_layers = torch.sqrt(eps_layers).detach().cpu().numpy().astype(complex)
        E_out = []
        H_out = []
        for ik in range(k.shape[0]):
            _, E_scnl, H_scnl = scattnlay.fieldnlay(
                (k[ik] * r_l).astype(float),
                (n_layers / n_env_np).astype(complex),
                *(k[ik] * r_probe_np).T,
                nmax=n_max_use,
            )
            E_out.append(torch.as_tensor(E_scnl, dtype=torch.complex128))
            H_out.append(torch.as_tensor(H_scnl, dtype=torch.complex128))
        E_out = torch.stack(E_out, dim=0)  # (Nk, Npos, 3)
        H_out = torch.stack(H_out, dim=0)
        return E_out, H_out

    E_base_ref, H_base_ref = _scattnlay_fields(r_layers_base, eps_layers_base)
    E_split_ref, H_split_ref = _scattnlay_fields(r_layers_split, eps_layers_split)

    # pmd shapes: (Np=1, Nk, Npos, 3)
    torch.testing.assert_close(
        base_nf["E_t"][0], E_base_ref, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        split_nf["E_t"][0], E_split_ref, rtol=1e-6, atol=1e-7
    )
