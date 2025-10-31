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


class TestNearfield(unittest.TestCase):
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
                r_core=r_core,
                r_shell=r_shell,
                mat_core=mat_core,
                mat_shell=mat_shell,
                mat_env=n_env,
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
            eps_c, eps_s, eps_env = p.get_material_permittivities(k0)

            fields_all = pmd.coreshell.nearfields(
                k0=k0,
                r_probe=r_probe,
                r_c=r_core,
                eps_c=eps_c,
                r_s=r_shell,
                eps_s=eps_s,
                eps_env=eps_env,
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
            H_scnl = np.nan_to_num(H_scnl).reshape(tuple(orig_shape_grid)) * Z0

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


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
