import importlib
import unittest
import torch
import numpy as np


# Skip the whole test case if the optional torchgdm dependency is not available.
_tg_spec = importlib.util.find_spec("torchgdm")
skip_if_no_tg = unittest.skipIf(_tg_spec is None, "torchgdm not installed")

# --------------------------------------------------------------------------- #
# Helper to build a simple pymiediff particle (core‑shell sphere)
# --------------------------------------------------------------------------- #
import pymiediff as pmd

try:
    import torchgdm as tg
except ModuleNotFoundError:
    tg = None  # type: ignore
    pass


def _make_particle():
    r_core = 100.0
    r_shell = 150.0
    n_core = 2.0
    n_shell = 1.5
    n_env = 1.0

    mat_core = pmd.materials.MatConstant(n_core**2)
    mat_shell = pmd.materials.MatConstant(n_shell**2)

    return pmd.Particle(
        r_core=r_core,
        r_shell=r_shell,
        mat_core=mat_core,
        mat_shell=mat_shell,
        mat_env=n_env,
    )


# --------------------------------------------------------------------------- #
# Test case
# --------------------------------------------------------------------------- #
@skip_if_no_tg
class TestTorchGDMStructs(unittest.TestCase):

    def setUp(self):
        self.particle = _make_particle()
        self.wavelengths = torch.tensor([600.0, 800.0], dtype=torch.float32)

        # Import classes lazily – they raise RuntimeError if torchgdm is missing.
        from pymiediff.helper.tg import (
            StructAutodiffMieEffPola3D,
            StructAutodiffMieGPM3D,
        )

        self.EffPolaCls = StructAutodiffMieEffPola3D
        self.GPMCls = StructAutodiffMieGPM3D

    # ----------------------------------------------------------------------- #
    # Point‑polarizability structure
    # ----------------------------------------------------------------------- #
    def test_struct_autodiff_mie_eff_pola_3d(self):
        struct = self.EffPolaCls(
            self.particle,
            wavelengths=self.wavelengths,
        )

        # 6×6 polarizability tensor per wavelength
        self.assertEqual(struct.alpha_data.shape, (1, len(self.wavelengths), 6, 6))

        # Position vector exists, has length 3 and lives on the same device as the particle
        self.assertTrue(hasattr(struct, "r0"))
        self.assertEqual(struct.r0.shape, (3,))

        self.assertEqual(str(struct.r0.device), str(self.particle.device))

    # ----------------------------------------------------------------------- #
    # Global‑polarizability‑matrix structure
    # ----------------------------------------------------------------------- #
    def test_struct_autodiff_mie_gpm_3d(self):
        # Use few probes / plane‑wave angles to keep CI fast.
        struct = self.GPMCls(
            self.particle,
            wavelengths=self.wavelengths,
            r_gpm=12,  # small number of GPM probe points
            n_src_pw_angles=4,  # fewer incident directions
            verbose=False,
            progress_bar=False,
        )

        # The GPM tensor is stored inside the first gpm_dict entry.
        gpm_dict = struct.gpm_dict[0]
        self.assertIn("GPM_N6xN6", gpm_dict)

        gpm = gpm_dict["GPM_N6xN6"]
        # Shape should be (N_wl, N_gpm*6, N_gpm*6)
        self.assertEqual(gpm.shape[0], len(self.wavelengths))
        self.assertEqual(gpm.shape[1], gpm_dict["n_gpm_dp"] * 6)
        self.assertEqual(gpm.shape[2], gpm_dict["n_gpm_dp"] * 6)


# ----------------------------------------------------------------------
# Skip the whole module if torchgdm (and thus the helper) is not installed
# ----------------------------------------------------------------------
@skip_if_no_tg
class TestTorchGDMeffDpvsMie(unittest.TestCase):
    """Compare torchgdm‑based Mie against the native pymiediff Mie solver.

    --> eff. polarizability version"""

    @staticmethod
    def _make_small_particle():
        """small core‑shell sphere"""
        r_core = 30.0  # nm
        r_shell = 40.0  # nm
        n_core = 2.0
        n_shell = 1.5
        n_env = 1.0

        mat_core = pmd.materials.MatConstant(n_core**2)
        mat_shell = pmd.materials.MatConstant(n_shell**2)

        return pmd.Particle(
            r_core=r_core,
            r_shell=r_shell,
            mat_core=mat_core,
            mat_shell=mat_shell,
            mat_env=n_env,
        )

    # ------------------------------------------------------------------
    # Extinction cross‑section
    # ------------------------------------------------------------------
    def _extinction_gdm(self, particle, wl):
        """Run a minimal torchgdm simulation and return the extinction CS."""
        import torchgdm as tg

        wl_tensor = torch.tensor([wl], dtype=torch.float32)

        struct = pmd.helper.tg.StructAutodiffMieEffPola3D(
            particle, wavelengths=wl_tensor
        )

        env = tg.env.freespace_3d.EnvHomogeneous3D(env_material=1.0)
        sim = tg.simulation.Simulation(
            structures=[struct],
            environment=env,
            illumination_fields=[
                tg.env.freespace_3d.PlaneWave(e0p=1.0, e0s=0.0, inc_angle=0.0)
            ],
            wavelengths=wl_tensor,
        )
        sim.run()
        cs = sim.get_spectra_crosssections()["ecs"][0].item()
        return cs

    def _extinction_mie(self, particle, wl):
        """Direct pymiediff Mie extinction cross‑section."""
        k0 = 2 * np.pi / wl
        cs = particle.get_cross_sections(k0=k0)["cs_ext"].item()
        return cs

    def test_extinction_cross_section(self):
        """Extinction cross‑section must agree within ~1 %."""
        import torchgdm as tg

        particle = self._make_small_particle()
        for wl in (550.0,):  # nm
            cs_gdm = self._extinction_gdm(particle, wl)
            cs_mie = self._extinction_mie(particle, wl)

            rel_err = abs(cs_gdm - cs_mie) / cs_mie
            self.assertLess(
                rel_err,
                0.015,
                msg=f"Extinction at {wl} nm differs by {rel_err:.2%}",
            )

    # ------------------------------------------------------------------
    # Near‑field intensity
    # ------------------------------------------------------------------
    def _nearfield_gdm(self, particle, wl, r_probe):
        """Run torchgdm and return scattered‑field intensity."""
        import torchgdm as tg

        wl_tensor = torch.tensor([wl], dtype=torch.float32)

        struct = pmd.helper.tg.StructAutodiffMieEffPola3D(
            particle, wavelengths=wl_tensor
        )

        env = tg.env.freespace_3d.EnvHomogeneous3D(env_material=1.0)
        sim = tg.simulation.Simulation(
            structures=[struct],
            environment=env,
            illumination_fields=[
                tg.env.freespace_3d.PlaneWave(e0p=1.0, e0s=0.0, inc_angle=0.0)
            ],
            wavelengths=wl_tensor,
        )
        sim.run()
        nf = sim.get_nearfield(wl, r_probe=r_probe)
        intensity = nf["sca"].get_efield_intensity()[0].cpu().numpy()
        return intensity

    def _nearfield_mie(self, particle, wl, r_probe):
        """Direct pymiediff near‑field evaluation."""
        k0 = 2 * np.pi / wl
        nf = particle.get_nearfields(k0=k0, r_probe=r_probe.cpu().numpy())
        intensity = torch.sum(torch.abs(nf["E_s"] ** 2), axis=-1).cpu().numpy()
        return intensity

    def test_nearfield_intensity(self):
        """Near‑field intensity must agree within ~1 % for several probe points."""
        import torchgdm as tg

        particle = self._make_small_particle()
        # probe points: a small square grid around the particle
        r_probe = tg.tools.geometry.coordinate_map_2d_square(
            d=250.0, n=5, r3=500.0, projection="xz"
        )["r_probe"]

        for wl in (550.0,):  # nm
            intensity_gdm = self._nearfield_gdm(particle, wl, r_probe)
            intensity_mie = self._nearfield_mie(particle, wl, r_probe)

            # point‑wise relative error, protect against division by zero
            denom = np.maximum(intensity_mie, 1e-12)
            rel_err = np.abs(intensity_gdm - intensity_mie) / denom
            max_err = np.max(rel_err)

            self.assertLess(
                max_err,
                0.015,
                msg=f"Near‑field at {wl} nm deviates up to {max_err:.2%}",
            )


@skip_if_no_tg
class TestTorchGDMeffGPMvsMie(unittest.TestCase):
    """Compare torchgdm‑based Mie against the native pymiediff Mie solver.

    --> GPM version"""

    @staticmethod
    def _make_small_particle():
        """small core‑shell sphere"""
        r_core = 30.0  # nm
        r_shell = 40.0  # nm
        n_core = 2.0
        n_shell = 1.5
        n_env = 1.0

        mat_core = pmd.materials.MatConstant(n_core**2)
        mat_shell = pmd.materials.MatConstant(n_shell**2)

        return pmd.Particle(
            r_core=r_core,
            r_shell=r_shell,
            mat_core=mat_core,
            mat_shell=mat_shell,
            mat_env=n_env,
        )

    # ------------------------------------------------------------------
    # Extinction cross‑section
    # ------------------------------------------------------------------
    def _extinction_gdm(self, particle, wl):
        """Run a minimal torchgdm simulation and return the extinction CS."""
        import torchgdm as tg

        wl_tensor = torch.tensor([wl], dtype=torch.float32)

        struct = pmd.helper.tg.StructAutodiffMieGPM3D(
            particle, wavelengths=wl_tensor, r_gpm=36
        )

        env = tg.env.freespace_3d.EnvHomogeneous3D(env_material=1.0)
        sim = tg.simulation.Simulation(
            structures=[struct],
            environment=env,
            illumination_fields=[
                tg.env.freespace_3d.PlaneWave(e0p=1.0, e0s=0.0, inc_angle=0.0)
            ],
            wavelengths=wl_tensor,
        )
        sim.run()
        cs = sim.get_spectra_crosssections()["ecs"][0].item()
        return cs

    def _extinction_mie(self, particle, wl):
        """Direct pymiediff Mie extinction cross‑section."""
        k0 = 2 * np.pi / wl
        cs = particle.get_cross_sections(k0=k0)["cs_ext"].item()
        return cs

    def test_extinction_cross_section(self):
        """Extinction cross‑section must agree within ~1 %."""
        import torchgdm as tg

        particle = self._make_small_particle()
        for wl in (550.0,):  # nm
            cs_gdm = self._extinction_gdm(particle, wl)
            cs_mie = self._extinction_mie(particle, wl)

            rel_err = abs(cs_gdm - cs_mie) / cs_mie
            self.assertLess(
                rel_err,
                0.015,
                msg=f"Extinction at {wl} nm differs by {rel_err:.2%}",
            )

    # ------------------------------------------------------------------
    # Near‑field intensity
    # ------------------------------------------------------------------
    def _nearfield_gdm(self, particle, wl, r_probe):
        """Run torchgdm and return scattered‑field intensity."""
        import torchgdm as tg

        wl_tensor = torch.tensor([wl], dtype=torch.float32)

        struct = pmd.helper.tg.StructAutodiffMieGPM3D(
            particle, wavelengths=wl_tensor, r_gpm=36
        )

        env = tg.env.freespace_3d.EnvHomogeneous3D(env_material=1.0)
        sim = tg.simulation.Simulation(
            structures=[struct],
            environment=env,
            illumination_fields=[
                tg.env.freespace_3d.PlaneWave(e0p=1.0, e0s=0.0, inc_angle=0.0)
            ],
            wavelengths=wl_tensor,
        )
        sim.run()
        nf = sim.get_nearfield(wl, r_probe=r_probe)
        intensity = nf["sca"].get_efield_intensity()[0].cpu().numpy()
        return intensity

    def _nearfield_mie(self, particle, wl, r_probe):
        """Direct pymiediff near‑field evaluation."""
        k0 = 2 * np.pi / wl
        nf = particle.get_nearfields(k0=k0, r_probe=r_probe.cpu().numpy())
        intensity = torch.sum(torch.abs(nf["E_s"] ** 2), axis=-1).cpu().numpy()
        return intensity

    def test_nearfield_intensity(self):
        """Near‑field intensity must agree within ~1 % for several probe points."""
        import torchgdm as tg

        particle = self._make_small_particle()
        # probe points: a small square grid around the particle
        r_probe = tg.tools.geometry.coordinate_map_2d_square(
            d=250.0, n=5, r3=500.0, projection="xz"
        )["r_probe"]

        for wl in (550.0,):  # nm
            intensity_gdm = self._nearfield_gdm(particle, wl, r_probe)
            intensity_mie = self._nearfield_mie(particle, wl, r_probe)

            # point‑wise relative error, protect against division by zero
            denom = np.maximum(intensity_mie, 1e-12)
            rel_err = np.abs(intensity_gdm - intensity_mie) / denom
            max_err = np.max(rel_err)

            self.assertLess(
                max_err,
                0.015,
                msg=f"Near‑field at {wl} nm deviates up to {max_err:.2%}",
            )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
