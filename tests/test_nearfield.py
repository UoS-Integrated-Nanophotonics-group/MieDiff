"""
Fully-differentiable scattered near-field
"""

# %%
import torch
import unittest
import warnings

import numpy as np
import torch
import pymiediff as pmd


class TestNearfieldScattnlay(unittest.TestCase):
    def test_vs_scattnlay(self):
        n_max = 10
        # Mie from `scattnlay` (includes internal fiels)
        # "pip install scattnlay"
        # https://github.com/ovidiopr/scattnlay
        try:
            from scattnlay import fieldnlay
        except ImportError:
            warnings.warn(
                "`scattnlay` seems not installed. "
                + "Skipping nearfield test vs `scattnlay`."
            )
            return

        for i_test_case in range(4):
            # - config test cases
            if i_test_case == 0:
                wl0 = torch.as_tensor([550])
                r_core = 50
                r_shell = r_core + 150
                n_core = 2.5
                n_shell = 4.50
                n_env = 1.0

            elif i_test_case == 1:
                wl0 = torch.as_tensor([650])
                r_core = 150
                r_shell = r_core + 200
                n_core = 4.5
                n_shell = 2.5
                n_env = 1.5

            elif i_test_case == 2:
                wl0 = torch.as_tensor([550])
                r_core = 200
                r_shell = r_core + 250
                n_core = 1.5
                n_shell = 1.5
                n_env = 1.5

            elif i_test_case == 3:
                wl0 = torch.as_tensor([600])
                r_core = 50
                r_shell = r_core + 80
                n_core = (-9.4277 + 1.5129j) ** 0.5
                n_shell = (15.4524 + 0.1456j) ** 0.5
                n_env = 1.0

            # --- pymiediff
            k0 = 2 * torch.pi / wl0
            mat_core = pmd.materials.MatConstant(n_core**2)
            mat_shell = pmd.materials.MatConstant(n_shell**2)

            # - setup the particle
            p = pmd.Particle(
                mat_env=n_env,
                r_core=r_core,
                mat_core=mat_core,
                r_shell=r_shell,
                mat_shell=mat_shell,
            )

            d_area_plot = 550
            x_offset = 0
            y_offset = 0
            z_offset = 0

            x, z = torch.meshgrid(
                torch.linspace(-d_area_plot, d_area_plot, 20),
                torch.linspace(-d_area_plot, d_area_plot, 20),
            )
            x = x + x_offset
            z = z + z_offset
            y = torch.ones_like(x) * y_offset
            grid = torch.stack([x, y, z], dim=-1)

            orig_shape_grid = grid.shape
            r_probe = grid.view(-1, 3)

            k0 = torch.as_tensor(k0, device=p.device)
            eps_layers, eps_env = p.get_material_permittivities(k0)

            fields_all = pmd.multishell.nearfields(
                k0=k0,
                r_probe=r_probe,
                r_layers=p.r_layers,
                eps_layers=eps_layers,
                eps_env=eps_env,
                backend="torch",
                n_max=n_max,
            )
            Etot = fields_all["E_t"]
            Htot = fields_all["H_t"]

            i_particle = 0
            i_wl = 0
            E_mie = Etot[i_particle, i_wl]
            E_mie = E_mie.view(orig_shape_grid)
            H_mie = Htot[i_particle, i_wl]
            H_mie = H_mie.view(orig_shape_grid)

            # --- scattnlayer
            k = k0 * n_env
            x = k.squeeze().numpy() * r_core
            y = k.squeeze().numpy() * r_shell
            m_c = n_core / n_env
            m_s = n_shell / n_env
            x_list = np.array([x, y])
            m_list = np.array([m_c, m_s])
            coords = r_probe

            terms, E_scnl, H_scnl = fieldnlay(
                x_list, m_list, *(k.squeeze().numpy() * coords.numpy()).T, nmax=n_max
            )

            E_scnl = np.nan_to_num(E_scnl).reshape(tuple(orig_shape_grid))
            Z0 = 376.73
            H_scnl = np.nan_to_num(H_scnl * n_env).reshape(tuple(orig_shape_grid)) * Z0

            E_mie[np.abs(E_scnl) > 100] = 0.0  # remove singularities
            H_mie[np.abs(H_scnl) > 100] = 0.0  # remove singularities
            E_scnl[np.abs(E_scnl) > 100] = 0.0  # remove singularities
            H_scnl[np.abs(H_scnl) > 100] = 0.0  # remove singularities

            # --- compare results
            # Note: `scattnlay` often produces inf or zero in corner pixels
            # It is not clear why this happens
            # to avoid this problem, we remove the outer pixels for the test.
            np.testing.assert_allclose(
                E_scnl[1:-1], E_mie[1:-1].numpy(), atol=5e-5, rtol=5e-5
            )
            np.testing.assert_allclose(
                H_scnl[1:-1], H_mie[1:-1].numpy(), atol=5e-5, rtol=5e-5
            )


class TestNearfieldComparison(unittest.TestCase):

    def _run_nearfield_comparison(self):
        # Helper function that runs the full near‑field comparison and returns
        # the four quantities needed for the assertions.

        # Mie from `treams` (only external fields)
        # "pip install treams"
        # https://github.com/tfp-photonics/treams
        try:
            import treams
        except ImportError:
            warnings.warn(
                "`treams` seems not installed. "
                + "Skipping nearfield test vs `treams`."
            )
            return

        # ----- configuration
        wl0 = torch.as_tensor([600.0])
        k0 = 2 * torch.pi / wl0

        r_core = 150
        r_shell = r_core + 150
        n_core = 2.5
        n_shell = 3.5
        n_env = 1.5

        mat_core = pmd.materials.MatConstant(n_core**2)
        mat_shell = pmd.materials.MatConstant(n_shell**2)

        p = pmd.Particle(
            mat_env=n_env,
            r_core=r_core,
            mat_core=mat_core,
            r_shell=r_shell,
            mat_shell=mat_shell,
        )

        # ----- probe grid (small grid sufficient for test)
        d_area_plot = 200
        x_offset = y_offset = z_offset = 0
        x, z = torch.meshgrid(
            torch.linspace(-d_area_plot, d_area_plot, 10),
            torch.linspace(-d_area_plot, d_area_plot, 10),
        )
        y = torch.full_like(x, 350.0)
        grid = torch.stack([x + x_offset, y + y_offset, z + z_offset], dim=-1)
        r_probe = grid.reshape(-1, 3)

        # ----- Mie coefficients
        k0 = k0.to(p.device)
        eps_layers, eps_env = p.get_material_permittivities(k0)
        r_layers = p.r_layers

        miecoeff = pmd.multishell.mie_coefficients(
            k0=k0,
            r_layers=r_layers,
            eps_layers=eps_layers,
            eps_env=eps_env,
            backend="torch",
            precision="double",
            which_jn="recurrence",
            n_max=10,
        )

        # ----- pymiediff near‑field
        fields_all = pmd.multishell.nearfields(
            k0=k0,
            r_probe=r_probe,
            r_layers=r_layers,
            eps_layers=eps_layers,
            eps_env=eps_env,
            backend="torch",
            n_max=miecoeff["n_max"],
        )
        Es_xyz = fields_all["E_t"]
        Hs_xyz = fields_all["H_t"]

        # ----- treams near‑field
        eps_spheres = [n_core**2, n_shell**2]
        radii = p.r_layers.detach().cpu().tolist()
        materials = [treams.Material(_eps) for _eps in eps_spheres] + [
            treams.Material(eps_env.item().real)
        ]
        lmax = 15
        sphere = treams.TMatrix.sphere(lmax, k0.item(), radii, materials)
        tm = treams.TMatrix.cluster([sphere], [[0, 0, 0]]).interaction.solve()
        inc = treams.plane_wave(
            [0, 0, k0.item()], pol=[1, 0, 0], k0=tm.k0, material=tm.material
        )
        sca = tm @ inc.expand(tm.basis)

        e_sca = sca.efield(grid.numpy())
        h_sca = sca.hfield(grid.numpy())
        e_inc = inc.efield(grid.numpy())
        h_inc = inc.hfield(grid.numpy())

        # mask interior of the shell (treams cannot do this)
        mask = (grid**2).sum(-1).numpy() < r_shell**2
        e_sca[mask] = 0.0
        h_sca[mask] = 0.0

        e_tot = e_inc + e_sca
        h_tot = h_inc + h_sca

        # reshape pymiediff results to the grid shape for easy comparison
        orig_shape = grid.shape[:-1]
        Es = Es_xyz[0, 0].reshape(*orig_shape, 3)
        Hs = Hs_xyz[0, 0].reshape(*orig_shape, 3)

        return Es, Hs, e_tot, h_tot

    def test_mie_vs_treams(self, tol=1e-4):
        """Run the full calculation and compare pymiediff with treams."""

        Es, Hs, e_tot, h_tot = self._run_nearfield_comparison()

        # intensity comparisons
        intensity_mie_E = np.sum(np.abs(Es.numpy()) ** 2, axis=-1)
        intensity_treams_E = np.sum(np.abs(e_tot) ** 2, axis=-1)
        intensity_mie_H = np.sum(np.abs(Hs.numpy()) ** 2, axis=-1)
        intensity_treams_H = np.sum(np.abs(h_tot) ** 2, axis=-1)

        np.testing.assert_allclose(
            intensity_mie_E, intensity_treams_E, rtol=tol, atol=tol
        )
        np.testing.assert_allclose(
            intensity_mie_H, intensity_treams_H, rtol=tol, atol=tol
        )

        # field‑by‑field comparisons
        np.testing.assert_allclose(Es.numpy(), e_tot, rtol=tol, atol=tol)
        np.testing.assert_allclose(Hs.numpy(), h_tot, rtol=tol, atol=tol)


class TestNearfieldMultilayerPena(unittest.TestCase):
    def test_autodiff_multilayer_nearfields(self):
        # verify gradients through internal + external nearfields for 3-layer sphere
        wl0 = torch.as_tensor([700.0], dtype=torch.float64)
        k0 = 2 * torch.pi / wl0
        n_env = 1.2

        r_layers = torch.tensor([45.0, 80.0, 120.0], dtype=torch.float64)
        n_layers = torch.tensor(
            [2.0 + 0.05j, 1.7 + 0.0j, 1.35 + 0.02j], dtype=torch.complex128
        )
        eps_layers = n_layers**2

        # include internal (r<r1) and external (r>r3) points
        r_probe = torch.tensor(
            [
                [10.0, 0.0, 0.0],   # internal
                [30.0, 0.0, 0.0],   # internal
                [0.0, 20.0, 10.0],  # internal off-axis
                [150.0, 0.0, 0.0],  # external
                [0.0, 0.0, 170.0],  # external on-axis
                [120.0, 60.0, 90.0],  # external off-axis
            ],
            dtype=torch.float64,
        )

        k0 = k0.clone().requires_grad_(True)
        eps_layers = eps_layers.clone().requires_grad_(True)

        res = pmd.multishell.nearfields(
            k0=k0,
            r_probe=r_probe,
            r_layers=r_layers,
            eps_layers=eps_layers,
            eps_env=n_env**2,
            backend="pena",
            n_max=50,
        )

        # use total field intensity to ensure non-trivial gradients
        E_t = res["E_t"]
        loss = (E_t.real**2 + E_t.imag**2).sum()

        loss.backward()

        # gradients should be finite and non-zero
        self.assertIsNotNone(k0.grad)
        self.assertIsNotNone(eps_layers.grad)

        self.assertTrue(torch.isfinite(k0.grad).all().item())
        self.assertTrue(torch.isfinite(eps_layers.grad).all().item())

        self.assertGreater(torch.abs(k0.grad).max().item(), 0.0)
        self.assertGreater(torch.abs(eps_layers.grad).max().item(), 0.0)

    def test_vs_scattnlay_internal_fields_layers(self):
        try:
            from scattnlay import fieldnlay
        except ImportError:
            warnings.warn("`scattnlay` seems not installed. Skipping multilayer test.")
            return

        Z0 = 376.73
        n_max = 50

        cases = [
            dict(
                name="homogeneous",
                wl0=torch.as_tensor([620.0], dtype=torch.float64),
                n_env=1.0,
                r_layers=torch.tensor([120.0], dtype=torch.float64),
                n_layers=torch.tensor([2.2 + 0.05j], dtype=torch.complex128),
            ),
            dict(
                name="two-layer",
                wl0=torch.as_tensor([700.0], dtype=torch.float64),
                n_env=1.3,
                r_layers=torch.tensor([60.0, 140.0], dtype=torch.float64),
                n_layers=torch.tensor(
                    [2.4 + 0.0j, 1.6 + 0.03j], dtype=torch.complex128
                ),
            ),
            dict(
                name="three-layer",
                wl0=torch.as_tensor([780.0], dtype=torch.float64),
                n_env=1.1,
                r_layers=torch.tensor([45.0, 90.0, 150.0], dtype=torch.float64),
                n_layers=torch.tensor(
                    [2.1 + 0.1j, 1.7 + 0.0j, 1.35 + 0.02j],
                    dtype=torch.complex128,
                ),
            ),
        ]

        for cfg in cases:
            wl0 = cfg["wl0"]
            k0 = 2 * torch.pi / wl0
            n_env = cfg["n_env"]
            r_layers = cfg["r_layers"]
            n_layers = cfg["n_layers"]
            eps_layers = n_layers**2

            # probe points: one inside each layer + one outside + a few off-axis points
            r_points = [0.4 * r_layers[0]]
            for li in range(1, len(r_layers)):
                r_mid = 0.5 * (r_layers[li - 1] + r_layers[li])
                r_points.append(r_mid)
            r_points.append(1.2 * r_layers[-1])

            r_probe_list = []
            for rr in r_points:
                r_probe_list.append([float(rr), 0.0, 0.0])
                r_probe_list.append([0.0, float(rr), 0.0])
                r_probe_list.append([0.0, 0.0, float(rr)])
            r_probe_list.append([0.0, 0.0, float(1.6 * r_layers[-1])])
            r_probe = torch.tensor(r_probe_list, dtype=torch.float64)

            res = pmd.multishell.nearfields(
                k0=k0,
                r_probe=r_probe,
                r_layers=r_layers,
                eps_layers=eps_layers,
                eps_env=n_env**2,
                backend="pena",
                n_max=n_max,
            )
            E_pmd = res["E_t"][0, 0].detach().cpu().numpy()
            H_pmd = res["H_t"][0, 0].detach().cpu().numpy()

            k = float((k0 * n_env).item())
            x_list = (k * r_layers.detach().cpu().numpy()).astype(np.float64)
            m_list = (n_layers.detach().cpu().numpy() / n_env).astype(np.complex128)
            _, E_scnl, H_scnl = fieldnlay(
                x_list,
                m_list,
                *(k * r_probe.detach().cpu().numpy()).T,
                nmax=n_max,
            )
            E_scnl = np.nan_to_num(E_scnl)
            H_scnl = np.nan_to_num(H_scnl * n_env) * Z0

            # filter out occasional scattnlay singularities
            mask = np.isfinite(E_scnl).all(axis=-1)
            mask &= np.isfinite(H_scnl).all(axis=-1)
            mask &= np.abs(E_scnl).max(axis=-1) < 100
            mask &= np.abs(H_scnl).max(axis=-1) < 100

            np.testing.assert_allclose(
                E_pmd[mask], E_scnl[mask], atol=7e-5, rtol=7e-5
            )
            np.testing.assert_allclose(
                H_pmd[mask], H_scnl[mask], atol=7e-5, rtol=7e-5
            )

    def test_vs_scattnlay_multilayer(self):
        try:
            from scattnlay import fieldnlay
        except ImportError:
            warnings.warn("`scattnlay` seems not installed. Skipping multilayer test.")
            return

        wl0 = torch.as_tensor([700.0], dtype=torch.float64)
        k0 = 2 * torch.pi / wl0
        n_env = 1.2

        r_layers = torch.tensor([45.0, 80.0, 130.0], dtype=torch.float64)
        n_layers = torch.tensor(
            [2.0 + 0.1j, 1.6 + 0.0j, 1.35 + 0.05j], dtype=torch.complex128
        )
        eps_layers = n_layers**2

        r_probe = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [50.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [170.0, 0.0, 0.0],
                [0.0, 60.0, 150.0],
                [150.0, 80.0, 120.0],
            ],
            dtype=torch.float64,
        )

        res = pmd.multishell.nearfields(
            k0=k0,
            r_probe=r_probe,
            r_layers=r_layers,
            eps_layers=eps_layers,
            eps_env=n_env**2,
            backend="pena",
            n_max=40,
        )
        E_pmd = res["E_t"][0, 0].detach().cpu().numpy()

        k = (k0 * n_env).item()
        x_list = (k * r_layers.detach().cpu().numpy()).astype(np.float64)
        m_list = (n_layers.detach().cpu().numpy() / n_env).astype(np.complex128)
        _, E_scnl, _ = fieldnlay(
            x_list,
            m_list,
            *(k * r_probe.detach().cpu().numpy()).T,
            nmax=40,
        )
        E_scnl = np.nan_to_num(E_scnl)

        np.testing.assert_allclose(E_pmd, E_scnl, atol=5e-5, rtol=5e-5)

    def test_vs_scattnlay_multilayer_grid(self):
        try:
            from scattnlay import fieldnlay
        except ImportError:
            warnings.warn("`scattnlay` seems not installed. Skipping multilayer test.")
            return

        wl0 = torch.as_tensor([700.0], dtype=torch.float64)
        k0 = 2 * torch.pi / wl0
        n_env = 1.2
        r_layers = torch.tensor([45.0, 80.0, 120.0, 160.0], dtype=torch.float64)
        n_layers = torch.tensor(
            [2.0 + 0.1j, 1.7 + 0.0j, 1.45 + 0.05j, 1.3 + 0.0j], dtype=torch.complex128
        )
        eps_layers = n_layers**2

        x, z = torch.meshgrid(
            torch.linspace(-220.0, 220.0, 21),
            torch.linspace(-220.0, 220.0, 21),
            indexing="ij",
        )
        y = torch.zeros_like(x)
        r_probe = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        res = pmd.multishell.nearfields(
            k0=k0,
            r_probe=r_probe,
            r_layers=r_layers,
            eps_layers=eps_layers,
            eps_env=n_env**2,
            backend="pena",
            n_max=70,
        )
        E_pmd = res["E_t"][0, 0].detach().cpu().numpy()

        k = float((k0 * n_env).item())
        x_list = (k * r_layers.detach().cpu().numpy()).astype(np.float64)
        m_list = (n_layers.detach().cpu().numpy() / n_env).astype(np.complex128)
        _, E_scnl, _ = fieldnlay(
            x_list,
            m_list,
            *(k * r_probe.detach().cpu().numpy()).T,
            nmax=70,
        )
        E_scnl = np.nan_to_num(E_scnl)

        r_norm = torch.linalg.norm(r_probe, dim=-1).detach().cpu().numpy()
        # Exclude the symmetry axis where phi is ill-defined and tiny spurious
        # components can appear after spherical->Cartesian transforms.
        rho = torch.linalg.norm(r_probe[:, :2], dim=-1).detach().cpu().numpy()
        mask = (r_norm > 1e-12) & (rho > 1e-10)
        np.testing.assert_allclose(E_pmd[mask], E_scnl[mask], atol=7e-5, rtol=7e-5)

    def test_multilayer_internal_external_field_decomposition(self):
        try:
            from scattnlay import fieldnlay
        except ImportError:
            warnings.warn("`scattnlay` seems not installed. Skipping multilayer test.")
            return

        wl0 = torch.as_tensor([700.0], dtype=torch.float64)
        k0 = 2 * torch.pi / wl0
        n_env = 1.2
        Z0 = 376.73

        r_layers = torch.tensor([45.0, 80.0, 120.0, 160.0], dtype=torch.float64)
        n_layers = torch.tensor(
            [2.0 + 0.1j, 1.7 + 0.0j, 1.45 + 0.05j, 1.3 + 0.0j], dtype=torch.complex128
        )
        eps_layers = n_layers**2

        # 3 internal + 3 external points
        r_probe = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [55.0, 0.0, 0.0],
                [105.0, 0.0, 0.0],
                [190.0, 0.0, 0.0],
                [0.0, 0.0, 220.0],
                [150.0, 80.0, 160.0],
            ],
            dtype=torch.float64,
        )

        res = pmd.multishell.nearfields(
            k0=k0,
            r_probe=r_probe,
            r_layers=r_layers,
            eps_layers=eps_layers,
            eps_env=n_env**2,
            backend="pena",
            n_max=70,
        )
        E_t = res["E_t"][0, 0].detach().cpu().numpy()
        H_t = res["H_t"][0, 0].detach().cpu().numpy()
        E_i = res["E_i"][0, 0].detach().cpu().numpy()
        H_i = res["H_i"][0, 0].detach().cpu().numpy()
        E_s = res["E_s"][0, 0].detach().cpu().numpy()
        H_s = res["H_s"][0, 0].detach().cpu().numpy()

        k = float((k0 * n_env).item())
        x_list = (k * r_layers.detach().cpu().numpy()).astype(np.float64)
        m_list = (n_layers.detach().cpu().numpy() / n_env).astype(np.complex128)
        _, E_ref, H_ref = fieldnlay(
            x_list,
            m_list,
            *(k * r_probe.detach().cpu().numpy()).T,
            nmax=70,
        )
        E_ref = np.nan_to_num(E_ref)
        H_ref = np.nan_to_num(H_ref * n_env) * Z0

        np.testing.assert_allclose(E_t, E_ref, atol=7e-5, rtol=7e-5)
        np.testing.assert_allclose(H_t, H_ref, atol=7e-5, rtol=7e-5)

        r_norm = np.linalg.norm(r_probe.detach().cpu().numpy(), axis=1)
        outside = r_norm > float(r_layers[-1].item())
        inside = ~outside

        # outside: total = incident + scattered
        np.testing.assert_allclose(
            E_t[outside], E_i[outside] + E_s[outside], atol=7e-5, rtol=7e-5
        )
        np.testing.assert_allclose(
            H_t[outside], H_i[outside] + H_s[outside], atol=7e-5, rtol=7e-5
        )

        # inside: scattered slot carries full internal field
        np.testing.assert_allclose(E_t[inside], E_s[inside], atol=7e-5, rtol=7e-5)
        np.testing.assert_allclose(H_t[inside], H_s[inside], atol=7e-5, rtol=7e-5)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
