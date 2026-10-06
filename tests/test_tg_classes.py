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
        mat_env=n_env,
        r_core=r_core,
        mat_core=mat_core,
        r_shell=r_shell,
        mat_shell=mat_shell,
    )


def _make_multilayer_particle():
    r_layers = torch.tensor([35.0, 55.0, 75.0, 95.0], dtype=torch.float64)
    mat_layers = [
        pmd.materials.MatConstant(2.0**2),
        pmd.materials.MatConstant(1.8**2),
        pmd.materials.MatConstant(1.6**2),
        pmd.materials.MatConstant(1.4**2),
    ]
    return pmd.Particle(
        r_layers=r_layers,
        mat_layers=mat_layers,
        mat_env=1.0,
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
            verbose=False,
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

    def test_struct_autodiff_mie_eff_pola_3d_multilayer(self):
        particle = _make_multilayer_particle()
        wavelengths = torch.tensor([700.0], dtype=torch.float32)
        struct = self.EffPolaCls(
            particle,
            wavelengths=wavelengths,
            verbose=False,
        )
        self.assertEqual(struct.alpha_data.shape, (1, len(wavelengths), 6, 6))


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
            mat_env=n_env,
            r_core=r_core,
            mat_core=mat_core,
            r_shell=r_shell,
            mat_shell=mat_shell,
        )

    # ------------------------------------------------------------------
    # Extinction cross‑section
    # ------------------------------------------------------------------
    def _extinction_gdm(self, particle, wl):
        """Run a minimal torchgdm simulation and return the extinction CS."""
        import torchgdm as tg

        wl_tensor = torch.tensor([wl], dtype=torch.float32)

        struct = pmd.helper.tg.StructAutodiffMieEffPola3D(
            particle, wavelengths=wl_tensor, verbose=False
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
        sim.run(verbose=False, progress_bar=False)
        cs = sim.get_spectra_crosssections(progress_bar=False)["ecs"][0].item()
        return cs

    def _extinction_mie(self, particle, wl):
        """Direct pymiediff Mie extinction cross‑section."""
        k0 = 2 * np.pi / wl
        cs = particle.get_cross_sections(k0=k0, backend="torch")["cs_ext"].item()
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
            particle, wavelengths=wl_tensor, verbose=False
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
        sim.run(verbose=False, progress_bar=False)
        nf = sim.get_nearfield(wl, r_probe=r_probe, progress_bar=False)
        intensity = nf["sca"].get_efield_intensity()[0].cpu().numpy()
        return intensity

    def _nearfield_mie(self, particle, wl, r_probe):
        """Direct pymiediff near‑field evaluation."""
        k0 = 2 * np.pi / wl
        nf = particle.get_nearfields(k0=k0, r_probe=r_probe.cpu().numpy(), backend="torch")
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
            mat_env=n_env,
            r_core=r_core,
            mat_core=mat_core,
            r_shell=r_shell,
            mat_shell=mat_shell,
        )

    # ------------------------------------------------------------------
    # Extinction cross‑section
    # ------------------------------------------------------------------
    def _extinction_gdm(self, particle, wl):
        """Run a minimal torchgdm simulation and return the extinction CS."""
        import torchgdm as tg

        wl_tensor = torch.tensor([wl], dtype=torch.float32)

        struct = pmd.helper.tg.StructAutodiffMieGPM3D(
            particle, wavelengths=wl_tensor, r_gpm=36, verbose=False, progress_bar=False
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
        sim.run(verbose=False, progress_bar=False)
        cs = sim.get_spectra_crosssections(progress_bar=False)["ecs"][0].item()
        return cs

    def _extinction_mie(self, particle, wl):
        """Direct pymiediff Mie extinction cross‑section."""
        k0 = 2 * np.pi / wl
        cs = particle.get_cross_sections(k0=k0, backend="torch")["cs_ext"].item()
        return cs

    def test_extinction_cross_section(self):
        """Extinction cross‑section must agree within ~1 %."""
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
            particle, wavelengths=wl_tensor, r_gpm=36, verbose=False, progress_bar=False
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
        sim.run(verbose=False, progress_bar=False)
        nf = sim.get_nearfield(wl, r_probe=r_probe, progress_bar=False)
        intensity = nf["sca"].get_efield_intensity()[0].cpu().numpy()
        return intensity

    def _nearfield_mie(self, particle, wl, r_probe):
        """Direct pymiediff near‑field evaluation."""
        k0 = 2 * np.pi / wl
        nf = particle.get_nearfields(k0=k0, r_probe=r_probe.cpu().numpy(), backend="torch")
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


# ----------------------------------------------------------------------
# torchgdm API-compatibility regression tests
# ----------------------------------------------------------------------
@skip_if_no_tg
class TestTorchGDMApiCompat(unittest.TestCase):
    """Guard against the torchgdm internals pymiediff depends on moving.

    torchgdm 0.58 moved the GPM tools from ``struct.eff_model_tools`` to
    ``struct.gpm_tools``, which broke ``StructAutodiffMieGPM3D`` with a bare
    ``ModuleNotFoundError``. The autodiff monkeypatch silently stopped applying
    in the same release because ``_get_full_Gdotalpha`` left ``LinearSystemBase``.
    """

    _ACCEPTED_MODULES = (
        "torchgdm.struct.gpm_tools",
        "torchgdm.struct.eff_model_tools",
    )

    def test_gpm_tool_resolver_resolves(self):
        """The GPM extraction helper resolves from a known torchgdm location."""
        from pymiediff.helper.tg import _import_gpm_tool

        func = _import_gpm_tool("extract_gpm_from_fields")
        self.assertTrue(callable(func))
        module = getattr(func, "__module__", "") or ""
        self.assertTrue(
            module.startswith(self._ACCEPTED_MODULES),
            msg=f"unexpected provider module for extract_gpm_from_fields: {module}",
        )

    def test_gpm_tool_resolver_error_names_all_paths(self):
        """A missing helper must name every path tried, not just the last one."""
        from pymiediff.helper.tg import _import_gpm_tool

        with self.assertRaises(ImportError) as ctx:
            _import_gpm_tool("extract_gpm_from_definitely_not_there")
        msg = str(ctx.exception)
        for path in (
            "torchgdm.struct.gpm_tools",
            "torchgdm.struct.eff_model_tools",
            "extract_gpm_from_definitely_not_there",
        ):
            self.assertIn(path, msg)

    def test_patch_torchgdm_autodiff_actually_applies(self):
        """The autodiff patch must land on the classes that define the method.

        It used to assign onto ``LinearSystemBase``, which stopped defining
        ``_get_full_Gdotalpha`` in torchgdm 0.58. Such a patch is a silent no-op,
        so assert on the class ``__dict__`` rather than mere attribute existence.
        """
        import torchgdm.linearsystem as ls
        from pymiediff.helper.tg import patch_torchgdm_autodiff

        report = patch_torchgdm_autodiff()

        self.assertIsInstance(report, dict)
        self.assertEqual(
            report["_get_full_Gdotalpha"],
            sorted(report["_get_full_Gdotalpha"]),
        )
        self.assertGreater(
            len(report["_get_full_Gdotalpha"]),
            0,
            msg=f"autodiff patch was a no-op, report: {report}",
        )
        for cls_name in report["_get_full_Gdotalpha"]:
            func = getattr(ls, cls_name).__dict__["_get_full_Gdotalpha"]
            self.assertEqual(func.__name__, "_get_full_Gdotalpha_no_inplace")

    def test_patch_torchgdm_autodiff_skips_incompatible_signature(self):
        """Solver classes with a different signature must be left alone.

        torchgdm's deprecated ``_LinearSystemFullInverse`` defines a
        ``_get_full_Gdotalpha`` taking positions/polarizabilities/self_terms.
        Replacing it with the ``sim``-based implementation would corrupt it.
        """
        import inspect

        import torchgdm.linearsystem as ls
        from pymiediff.helper.tg import patch_torchgdm_autodiff

        report = patch_torchgdm_autodiff()
        patched = set(report["_get_full_Gdotalpha"])

        for cls_name in dir(ls):
            cls = getattr(ls, cls_name, None)
            if not isinstance(cls, type):
                continue
            func = cls.__dict__.get("_get_full_Gdotalpha")
            if func is None or func.__name__ == "_get_full_Gdotalpha_no_inplace":
                continue
            params = set(inspect.signature(func).parameters)
            self.assertFalse(
                {"sim", "G_func", "wavelength"} <= params,
                msg=f"{cls_name} defines a compatible signature but was not patched",
            )

    def test_gpm_struct_and_simulation_match_mie(self):
        """End-to-end: a GPM structure reproduces the native Mie extinction."""
        import torchgdm as tg
        from pymiediff.helper.tg import patch_torchgdm_autodiff

        # the repaired patch is now genuinely active, so exercise it here
        patch_torchgdm_autodiff()

        particle = self._make_small_particle()
        wl_tensor = torch.tensor([550.0], dtype=torch.float32)

        struct = pmd.helper.tg.StructAutodiffMieGPM3D(
            particle, wavelengths=wl_tensor, r_gpm=18, verbose=False, progress_bar=False
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
        sim.run(verbose=False, progress_bar=False)
        cs_gdm = sim.get_spectra_crosssections(progress_bar=False)["ecs"][0].item()

        cs_mie = particle.get_cross_sections(
            k0=2 * np.pi / 550.0, backend="torch"
        )["cs_ext"].item()
        rel_err = abs(cs_gdm - cs_mie) / cs_mie
        self.assertLess(rel_err, 0.015, msg=f"GPM extinction off by {rel_err:.2%}")

    @staticmethod
    def _make_small_particle():
        r_core = 30.0  # nm
        r_shell = 40.0  # nm
        return pmd.Particle(
            mat_env=1.0,
            r_core=r_core,
            mat_core=pmd.materials.MatConstant(2.0**2),
            r_shell=r_shell,
            mat_shell=pmd.materials.MatConstant(1.5**2),
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
