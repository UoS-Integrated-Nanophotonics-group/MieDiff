# -*- coding: utf-8 -*-
"""
pymiediff.helper.tg
===================

Utility module that bridges **pymiediff** with the optional **torchgdm** package.
It provides:

* Thin wrappers around the TorchGDM API for Mie‑theory based structures,
  exposing auto‑differentiable point‑polarizability (`StructAutodiffMieEffPola3D`)
  and global‑polarizability‑matrix (`StructAutodiffMieGPM3D`) classes.
* Functions to compute Mie scattering coefficients (`mie_ab_sphere_3d_AD`) and to
  extract a full GPM model from a `pymiediff.Particle` (`extract_GPM_sphere_miediff`).
* Helper routines for rotation matrices, plane‑wave configuration generation,
  and evaluation of near‑fields via `pymiediff` (`_eval_mie`).

The module lazily imports ``torchgdm``; if the package is missing the public
classes raise a clear ``RuntimeError`` with installation instructions.  All
functions are written to be fully **torch‑autograd** compatible, enabling gradient‑based
optimisation of particle parameters (radii, refractive indices, etc.) in downstream
simulations.

Typical usage
-------------
```python
import pymiediff as pmd
from pymiediff.helper.tg import StructAutodiffMieEffPola3D

# define a core‑shell particle
particle = pmd.Particle(...)
# create an auto‑diff point polarizability structure
struct = StructAutodiffMieEffPola3D(particle, wavelengths=torch.linspace(500, 800, 5))
```

If torchgdm is not installed, importing this module will still succeed,
but any attempt to instantiate the classes will raise the informative
_MissingDependency error.

"""

# %%
import importlib
import time
import warnings

import numpy as np
import pymiediff as pmd
import torch

_tg_available = importlib.util.find_spec("torchgdm")


if _tg_available is None:

    class _MissingDependency(RuntimeError):
        def __init__(self, *args, **kwargs):
            super().__init__(
                "The optional dependency 'torchgdm' is required for "
                "'pymiediff.helper.tg' Install it via `pip install torchgdm`."
            )

    StructAutodiffMieEffPola3D = _MissingDependency
    StructAutodiffMieGPM3D = _MissingDependency

else:
    from torchgdm.constants import DTYPE_COMPLEX, DTYPE_FLOAT
    import torchgdm as tg


def _resolve_mie_backend(mie_particle, backend):
    """Pick a compatible Mie backend from particle configuration.

    Parameters
    ----------
    mie_particle : pymiediff.Particle
        Particle instance.
    backend : str or None
        Explicit backend override.

    Returns
    -------
    str
        Backend name.
    """
    if backend is not None:
        return backend
    if getattr(mie_particle, "_use_layers", False):
        n_layers = int(mie_particle.r_layers.numel())
        if n_layers > 2:
            return "pena"
    return "torch"


def _get_env_eps(mie_particle, wavelengths, device):
    """Evaluate environment permittivity at given wavelengths.

    Parameters
    ----------
    mie_particle : pymiediff.Particle
        Particle with environment material definition.
    wavelengths : torch.Tensor
        Wavelengths in nm.
    device : str or torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Environment permittivity values.
    """
    eps_env = mie_particle.mat_env.get_epsilon(wavelength=wavelengths)
    return torch.atleast_1d(torch.as_tensor(eps_env, device=device))


def _get_enclosing_radius(mie_particle):
    """Return outermost particle radius in both API modes.

    Parameters
    ----------
    mie_particle : pymiediff.Particle
        Particle definition.

    Returns
    -------
    torch.Tensor
        Enclosing radius.
    """
    if getattr(mie_particle, "_use_layers", False):
        return mie_particle.r_layers[-1]
    if mie_particle.r_s is None:
        return mie_particle.r_c
    return mie_particle.r_s


# ------ eff. dipole pair extraction from pymiediff particle -------
def mie_ab_sphere_3d_AD(
    mie_particle,
    wavelengths: torch.Tensor,
    n_env=None,
    n_max=2,
    as_dict=False,
    backend=None,
):
    """Return Mie coefficients for a particle in torchgdm-ready layout.

    Parameters
    ----------
    mie_particle : pymiediff.Particle
        Particle definition.
    wavelengths : torch.Tensor
        Wavelengths in nm.
    n_env : float, optional
        Environment refractive index override.
    n_max : int, default=2
        Maximum Mie order.
    as_dict : bool, default=False
        If ``True``, return metadata in addition to coefficients.
    backend : str, optional
        Solver backend (``"torch"``, ``"scipy"``, ``"pena"``). If ``None``,
        selected automatically.

    Returns
    -------
    tuple or dict
        ``(a_n, b_n)`` or dictionary with coefficients and metadata.
    """
    from torchgdm.env import EnvHomogeneous3D

    # --- preparation, tensor conversion
    assert type(mie_particle) == pmd.Particle, "Requires pymiediff particle"
    device = mie_particle.device

    wavelengths = torch.as_tensor(wavelengths, dtype=DTYPE_FLOAT, device=device)
    wavelengths = torch.atleast_1d(wavelengths)
    k0 = 2 * torch.pi / wavelengths

    backend = _resolve_mie_backend(mie_particle, backend)
    eps_env = _get_env_eps(mie_particle, wavelengths, device)
    assert torch.all(eps_env == eps_env[0]), "dispersive environment not supported yet"
    if n_env is not None:
        assert eps_env[0] == n_env**2
        eps_env = n_env**2
    env_3d = EnvHomogeneous3D(env_material=float(eps_env[0].real), device=device)

    # --- get Mie coefficients from pymiediff particle
    miecoeff = mie_particle.get_mie_coefficients(k0=k0, n_max=n_max, backend=backend)

    a_n = miecoeff["a_n"]
    b_n = miecoeff["b_n"]

    if a_n.ndim == 1:  # single wl
        a_n = a_n.unsqueeze(1)  # add empty wavelength dimension
    if b_n.ndim == 1:
        b_n = b_n.unsqueeze(1)

    a_n = a_n.moveaxis(0, 1)
    b_n = b_n.moveaxis(0, 1)  # move Mie-order last

    # full radius
    r_enclosing = _get_enclosing_radius(mie_particle)

    if as_dict:
        return dict(
            a_n=a_n,
            b_n=b_n,
            environment=env_3d,
            n_env=eps_env**0.5,
            device=device,
            r_enclosing=r_enclosing,
            wavelengths=wavelengths,
        )
    else:
        return a_n, b_n


# ------------- GPM extraction from pymiediff particle -------------
def rotation_x(alpha, device="cpu", DTYPE_FLOAT=torch.float32):
    """Return clockwise 3D rotation matrix around x-axis.

    Parameters
    ----------
    alpha : float or torch.Tensor
        Rotation angle in radians.
    device : str or torch.device, default="cpu"
        Target tensor device.
    DTYPE_FLOAT : torch.dtype, default=torch.float32
        Output dtype.

    Returns
    -------
    torch.Tensor
        Rotation matrix of shape ``(3, 3)``.
    """
    alpha = torch.as_tensor(alpha, dtype=DTYPE_FLOAT, device=device)
    s = torch.sin(alpha)
    c = torch.cos(alpha)
    rot_x = torch.as_tensor(
        [[1, 0, 0], [0, c, -s], [0, s, c]],
        dtype=DTYPE_FLOAT,
        device=device,
    )
    return rot_x


def rotation_y(alpha, device="cpu", DTYPE_FLOAT=torch.float32):
    """Return clockwise 3D rotation matrix around y-axis.

    Parameters
    ----------
    alpha : float or torch.Tensor
        Rotation angle in radians.
    device : str or torch.device, default="cpu"
        Target tensor device.
    DTYPE_FLOAT : torch.dtype, default=torch.float32
        Output dtype.

    Returns
    -------
    torch.Tensor
        Rotation matrix of shape ``(3, 3)``.
    """
    alpha = torch.as_tensor(alpha, dtype=DTYPE_FLOAT, device=device)
    s = torch.sin(alpha)
    c = torch.cos(alpha)
    rot_y = torch.as_tensor(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        dtype=DTYPE_FLOAT,
        device=device,
    )
    return rot_y


def rotation_z(alpha, device="cpu", DTYPE_FLOAT=torch.float32):
    """Return clockwise 3D rotation matrix around z-axis.

    Parameters
    ----------
    alpha : float or torch.Tensor
        Rotation angle in radians.
    device : str or torch.device, default="cpu"
        Target tensor device.
    DTYPE_FLOAT : torch.dtype, default=torch.float32
        Output dtype.

    Returns
    -------
    torch.Tensor
        Rotation matrix of shape ``(3, 3)``.
    """
    alpha = torch.as_tensor(alpha, dtype=DTYPE_FLOAT, device=device)
    s = torch.sin(alpha)
    c = torch.cos(alpha)
    rot_z = torch.as_tensor(
        [[c, -s, 0], [s, c, 0], [0, 0, 1]],
        dtype=DTYPE_FLOAT,
        device=device,
    )
    return rot_z


def setup_plane_waves_configs(n_angles, inc_planes=["xz", "xy"]):
    """Create illumination configuration dictionaries for plane waves.

    Parameters
    ----------
    n_angles : int
        Number of in-plane incidence angles.
    inc_planes : list of str, default=["xz", "xy"]
        Incidence planes.

    Returns
    -------
    list of dict
        Illumination definitions with keys ``angle``, ``inc_plane``,
        and ``polarization``.
    """
    inc_fields_conf = []
    for angle in torch.linspace(0, 2 * torch.pi, n_angles + 1)[:-1]:
        for inc in inc_planes:
            for pol in ["s", "p"]:
                inc_fields_conf.append(
                    dict(angle=angle, inc_plane=inc, polarization=pol)
                )
    return inc_fields_conf

def _plane_wave_rotation(inc_angle, inc_plane, polarization, device):
    """Return the rotation from the native pymiediff source to a target wave.

    The native near-field backend corresponds to a +z-propagating plane wave
    with electric field along +x, i.e. a ``yz``-plane ``s`` polarization.
    """
    angle = torch.as_tensor(inc_angle, dtype=DTYPE_FLOAT, device=device)
    sin_a = torch.sin(angle)
    cos_a = torch.cos(angle)
    zero = torch.zeros((), dtype=DTYPE_FLOAT, device=device)
    one = torch.ones((), dtype=DTYPE_FLOAT, device=device)

    if inc_plane == "yz":
        k_dir = torch.stack([zero, sin_a, cos_a])
        if polarization == "s":
            e_dir = torch.stack([one, zero, zero])
        elif polarization == "p":
            e_dir = torch.stack([zero, -cos_a, sin_a])
        else:
            raise ValueError(f"unsupported polarization: {polarization}")
    elif inc_plane == "xz":
        k_dir = torch.stack([sin_a, zero, cos_a])
        if polarization == "s":
            e_dir = torch.stack([zero, one, zero])
        elif polarization == "p":
            e_dir = torch.stack([-cos_a, zero, sin_a])
        else:
            raise ValueError(f"unsupported polarization: {polarization}")
    elif inc_plane == "xy":
        k_dir = torch.stack([sin_a, cos_a, zero])
        if polarization == "s":
            e_dir = torch.stack([zero, zero, one])
        elif polarization == "p":
            e_dir = torch.stack([-cos_a, sin_a, zero])
        else:
            raise ValueError(f"unsupported polarization: {polarization}")
    else:
        raise ValueError(f"unsupported incidence plane: {inc_plane}")

    e_dir = e_dir / torch.linalg.norm(e_dir)
    k_dir = k_dir / torch.linalg.norm(k_dir)
    h_dir = -torch.linalg.cross(e_dir, k_dir)
    h_dir = h_dir / torch.linalg.norm(h_dir)
    frame = torch.stack([e_dir, h_dir, k_dir], dim=0)
    return frame.T


# - parallelized treams evaluation (illumination and scattering)
def _eval_mie(mie_particle, inc_conf, k0, r_probe, r_gpm, n_max=None, backend=None):
    """Evaluate near fields for one illumination setup.

    Parameters
    ----------
    mie_particle : pymiediff.Particle
        Particle model.
    inc_conf : dict
        Illumination config with keys ``angle``, ``inc_plane``, and
        ``polarization``.
    k0 : torch.Tensor
        Single vacuum wavevector value.
    r_probe : torch.Tensor
        Probe coordinates for scattered fields.
    r_gpm : torch.Tensor
        Probe coordinates for incident fields.
    n_max : int, optional
        Mie truncation order.
    backend : str, optional
        Backend override.

    Returns
    -------
    tuple of torch.Tensor
        ``(E_sca, H_sca, E_inc, H_inc)``.
    """
    # - calc incident and scattered fields
    pol_type = inc_conf["polarization"]
    inc_plane = inc_conf["inc_plane"]
    inc_angle = inc_conf["angle"]  # rad

    rot = _plane_wave_rotation(
        inc_angle=inc_angle,
        inc_plane=inc_plane,
        polarization=pol_type,
        device=mie_particle.device,
    )

    r_probe_rot = torch.matmul(r_probe, rot)
    r_gpm_rot = torch.matmul(r_gpm, rot)

    # caclulate nearfields with pymiediff
    backend = _resolve_mie_backend(mie_particle, backend)
    fields_sca = mie_particle.get_nearfields(
        k0,
        r_probe_rot,
        n_max=n_max,
        backend=backend,
    )
    fields_inc = mie_particle.get_nearfields(
        k0,
        r_gpm_rot,
        n_max=n_max,
        backend=backend,
    )

    # reverse grid rotation on fields
    rot_rev = rot.T.to(dtype=fields_sca["E_s"].dtype)
    for k in fields_sca:
        fields_sca[k] = torch.matmul(fields_sca[k], rot_rev)
    for k in fields_inc:
        fields_inc[k] = torch.matmul(fields_inc[k], rot_rev)

    e_sca = fields_sca["E_s"]
    h_sca = fields_sca["H_s"]
    e_inc = fields_inc["E_i"]
    h_inc = fields_inc["H_i"]

    return e_sca, h_sca, e_inc, h_inc


def extract_GPM_sphere_miediff(
    mie_particle,
    wavelengths,
    r_gpm,
    r_probe=None,
    n_src_pw_angles=12,
    r_probe_add=20,  # nm
    n_env=None,
    n_max=None,
    backend=None,
    verbose=True,
    progress_bar=True,
    **kwargs,
):
    """Extract a GPM model from ``pymiediff`` near fields.

    Parameters
    ----------
    mie_particle : pymiediff.Particle
        Source particle.
    wavelengths : torch.Tensor
        Wavelengths in nm.
    r_gpm : int, float, or torch.Tensor
        GPM support points or number of quasi-random support points to generate.
    r_probe : torch.Tensor, optional
        Probe coordinates for scattered fields.
    n_src_pw_angles : int, default=12
        Number of incidence angles per plane.
    r_probe_add : float, default=20
        Offset from enclosing radius for autogenerated ``r_probe``.
    n_env : float, optional
        Environment refractive index override.
    n_max : int, optional
        Mie truncation order.
    backend : str, optional
        Backend override.
    verbose : bool, default=True
        Print progress messages.
    progress_bar : bool, default=True
        Enable tqdm progress bar.
    **kwargs
        Forwarded to ``torchgdm.struct.eff_model_tools.extract_gpm_from_fields``.

    Returns
    -------
    dict
        GPM dictionary compatible with torchgdm structures.
    """
    DEFAULT_R_GPM = 36
    DEFAULT_R_PROBE_PHI = 36
    DEFAULT_R_PROBE_TETA = 18

    from torchgdm.env import EnvHomogeneous3D
    from torchgdm.struct.eff_model_tools import extract_gpm_from_fields

    # --- preparation, tensor conversion
    assert type(mie_particle) == pmd.Particle, "Requires pymiediff particle"
    device = mie_particle.device

    wavelengths = torch.as_tensor(wavelengths, dtype=DTYPE_FLOAT, device=device)
    wavelengths = torch.atleast_1d(wavelengths)
    k0 = 2 * torch.pi / wavelengths

    backend = _resolve_mie_backend(mie_particle, backend)
    eps_env = _get_env_eps(mie_particle, wavelengths, device)
    assert torch.all(eps_env == eps_env[0]), "dispersive environment not supported yet"
    if n_env is not None:
        assert eps_env[0] == n_env**2
        eps_env = n_env**2
    env_3d = EnvHomogeneous3D(env_material=float(eps_env[0].real), device=device)

    # full radius
    r_enclosing = _get_enclosing_radius(mie_particle)


    # --- gpm locations and extraction probe points
    if verbose:
        t0 = time.time()
        print("Extracting GPM model via pymiediff near-field eval...")

    if r_gpm is None:
        r_gpm = DEFAULT_R_GPM

    if type(r_gpm) in (int, float):
        from scipy.stats import qmc

        r_gpm = int(r_gpm)
        sampler = qmc.Halton(d=3, scramble=False)
        r_gpm = sampler.random(n=r_gpm)
        r_gpm /= np.max(np.linalg.norm(r_gpm, axis=1)) + 0.0001
        r_gpm -= np.mean(r_gpm, axis=0)
        r_gpm *= 0.65 * (float(r_enclosing) * 2.0)

    if r_probe is None:
        r_probe = tg.tools.geometry.coordinate_map_2d_spherical(
            r=r_enclosing + r_probe_add,
            n_phi=DEFAULT_R_PROBE_PHI,
            n_teta=DEFAULT_R_PROBE_TETA,
            device=mie_particle.device,
        )["r_probe"]

    r_probe = torch.as_tensor(r_probe, dtype=DTYPE_FLOAT, device=device)
    r_gpm = torch.as_tensor(r_gpm, dtype=DTYPE_FLOAT, device=device)

    # --- setup illumination sources (plane waves)
    if verbose:
        print("   setup illuminations...")

    inc_field_configs = setup_plane_waves_configs(
        n_src_pw_angles,
        inc_planes=["xz", "yz", "xy"],
    )

    # --- calc fields: scattering at r_probe; illuminations at r_gpm
    results = []
    for k0_single in tg.tqdm(k0, progress_bar=progress_bar, title="GPM extraction"):
        mie_results = [
            _eval_mie(
                mie_particle,
                inc_conf,
                k0_single,
                r_probe,
                r_gpm,
                n_max=n_max,
                backend=backend,
            )
            for inc_conf in inc_field_configs
        ]

        # - fill full arrays
        e_inc_mie = torch.zeros(
            (len(inc_field_configs), len(r_gpm), 3), dtype=DTYPE_COMPLEX
        )
        h_inc_mie = torch.zeros(
            (len(inc_field_configs), len(r_gpm), 3), dtype=DTYPE_COMPLEX
        )
        e_sca_mie = torch.zeros(
            (len(inc_field_configs), len(r_probe), 3), dtype=DTYPE_COMPLEX
        )
        h_sca_mie = torch.zeros(
            (len(inc_field_configs), len(r_probe), 3), dtype=DTYPE_COMPLEX
        )
        for i_inc, fields in enumerate(mie_results):
            norm = max(float(torch.abs(field).max().detach().cpu()) for field in fields)
            norm = max(norm, 1e-30)

            e_sca_mie[i_inc] = fields[0] / norm
            h_sca_mie[i_inc] = fields[1] / norm
            e_inc_mie[i_inc] = fields[2] / norm
            h_inc_mie[i_inc] = fields[3] / norm

        # - optimize GPM
        gpm_dict = extract_gpm_from_fields(
            wavelength=2 * torch.pi / k0_single,
            efields_sca=e_sca_mie.clone().contiguous(),
            hfields_sca=h_sca_mie.clone().contiguous(),
            efields_inc=e_inc_mie.clone().contiguous(),
            hfields_inc=h_inc_mie.clone().contiguous(),
            r_probe=r_probe,
            r_gpm=r_gpm,
            environment=env_3d,
            device=device,
            verbose=False,
            return_all_results=True,
            **kwargs,
        )
        results.append(gpm_dict["GPM"])

    # --- gather and return results
    GPM_N6xN6 = torch.stack(results, dim=0).to(dtype=DTYPE_COMPLEX, device=device)

    # dummy sphere geometry
    from torchgdm.struct.struct3d import sphere
    from torchgdm.struct.struct3d import discretizer_cubic

    _step_dummy_sphere = r_enclosing / 7
    _geo_dummy = discretizer_cubic(
        *sphere(r=r_enclosing / _step_dummy_sphere),
        step=_step_dummy_sphere,
        z_offset=0,
    )
    _geo_dummy = torch.as_tensor(_geo_dummy, dtype=DTYPE_FLOAT, device=device)

    env_3d.set_device(device)

    dict_gpm = dict(
        r_gpm=r_gpm,
        GPM_N6xN6=GPM_N6xN6,
        wavelengths=wavelengths,
        # additional metadata
        full_geometry=_geo_dummy,
        n_gpm_dp=len(r_gpm),
        r0=torch.as_tensor([0, 0, 0], dtype=DTYPE_FLOAT, device=device),
        enclosing_radius=r_enclosing,
        k0_spectrum=2 * torch.pi / wavelengths,
        environment=env_3d,
        extraction_r_probe=r_probe,
        mie_particle=mie_particle,
    )
    if verbose:
        print("Done in {:.2}s.".format(time.time() - t0))

    return dict_gpm


def combine_gpm_structures_autodiff(structures, environment=None, device=None):
    """Combine autodiff-capable GPM structures without in-place modification.

    This avoids inplace concatenation in torchgdm's `combine`, which can break
    autograd when structure parameters require gradients.
    """
    from torchgdm.struct.struct3d.gpm3d import StructGPM3D

    if not structures:
        raise ValueError("`structures` must contain at least one structure.")

    positions = torch.stack([s.r0 for s in structures], dim=0)
    gpm_dicts = []
    for s in structures:
        if isinstance(s.gpm_dict, list):
            if len(s.gpm_dict) != 1:
                raise ValueError("Expected single-entry gpm_dict for each structure.")
            gpm_dicts.append(s.gpm_dict[0])
        else:
            gpm_dicts.append(s.gpm_dict)

    if environment is None:
        environment = structures[0].environment
    if device is None:
        device = structures[0].device

    return StructGPM3D(
        positions=positions,
        gpm_dicts=gpm_dicts,
        environment=environment,
        device=device,
    )


def patch_torchgdm_autodiff():
    """Patch torchgdm linear system to avoid inplace ops that break autograd."""
    try:
        import torchgdm.linearsystem as ls
    except Exception:
        return

    def _get_full_Gdotalpha_no_inplace(self, sim, G_func, wavelength):
        all_alpha = []
        for s in sim.structures:
            all_alpha += list(
                s.get_polarizability_6x6(wavelength, sim.environment).unbind()
            )
        block_pola = torch.block_diag(*all_alpha)

        all_selfterms = []
        for s in sim.structures:
            all_selfterms += list(
                s.get_selfterm_6x6(wavelength, sim.environment).unbind()
            )
        block_selfterms = torch.block_diag(*all_selfterms)

        ones_gpm = torch.block_diag(
            *[torch.ones_like(s.real).to(torch.int) for s in all_alpha]
        )

        interact_NxNx6x6 = self._get_full_interaction_matrix_G_tensors(
            sim.get_all_positions(), G_func, wavelength
        )
        interact_NxN = ls._reduce_dimensions(interact_NxNx6x6)
        interact_NxN = interact_NxN.masked_fill(ones_gpm == 1, 0)
        interact_NxN = interact_NxN + block_selfterms

        return torch.matmul(interact_NxN, block_pola)

    ls.LinearSystemBase._get_full_Gdotalpha = _get_full_Gdotalpha_no_inplace

    def _zero_fill_nonpolarizable_fields_no_inplace(self, sim, field_at_polarizable):
        from torchgdm.constants import DTYPE_COMPLEX

        fields_full = torch.zeros(
            (len(sim.illumination_fields), len(sim.get_all_positions()), 6),
            dtype=DTYPE_COMPLEX,
            device=self.device,
        )

        mask_fields = sim._get_polarizable_mask_full_fields().view(-1)
        mask_idx = torch.nonzero(mask_fields, as_tuple=False).squeeze(-1)

        flat_full = fields_full.view(-1)
        flat_vals = field_at_polarizable.view(-1)
        flat_full = flat_full.scatter(0, mask_idx, flat_vals)
        return flat_full.view_as(fields_full)

    if hasattr(ls, "LinearSystemFullMemEff"):
        ls.LinearSystemFullMemEff._zero_fill_nonpolarizable_fields = (
            _zero_fill_nonpolarizable_fields_no_inplace
        )
    if hasattr(ls, "LinearSystemFullInverse"):
        ls.LinearSystemFullInverse._zero_fill_nonpolarizable_fields = (
            _zero_fill_nonpolarizable_fields_no_inplace
        )

    def _solve_no_inplace(self, sim, wavelength, batch_size=32, verbose=1):
        interact = self.get_interact(sim, wavelength, verbose=verbose)
        LU, pivots = torch.linalg.lu_factor(interact.clone())
        f0 = sim._get_polarizablefields_e0_h0(wavelength)
        f0 = f0.view(len(f0), -1, 1)
        f_masked = ls._batched_lu_solve(LU, pivots, f0=f0, batch_size=batch_size)
        eh_inside = self._zero_fill_nonpolarizable_fields(sim, f_masked[..., 0])
        e_inside, h_inside = torch.chunk(eh_inside, 2, dim=2)
        return e_inside, h_inside

    if hasattr(ls, "LinearSystemFullMemEff"):
        ls.LinearSystemFullMemEff.solve = _solve_no_inplace


# ---  torchgdm structure classes based on pymiediff Mie solver
if _tg_available is not None:

    class StructAutodiffMieEffPola3D(tg.struct.StructEffPola3D):
        """TorchGDM Mie-theory based 3D point polarizability with auto-diff support

        Requires `torchGDM` (pip install torchgdm)

        Defines a point polarizability representing a sphere using
        first order (dipolar) Mie coefficients
        """

        __name__ = "Mie-theory sphere dipolar polarizability (3D) structure class"

        def __init__(
            self,
            mie_particle,
            wavelengths: torch.Tensor,
            n_env=None,
            backend=None,
            r0: torch.Tensor = None,
            quadrupol_tol=0.15,
            verbose=True,
        ):
            """Build a dipolar 3D effective-polarizability structure.

            Parameters
            ----------
            mie_particle : :class:`pymiediff.Particle`
                The Mie particle to build a GPM model for.
            wavelengths : torch.Tensor
                Wavelengths (in nm) at which the polarizability is evaluated.
            n_env : float, optional
                Refractive index of the surrounding medium.
                If not provided, use the environment defined in the particle.
            r0 : torch.Tensor, optional
                Position of the point polarizability (x, y, z).  If ``None``,
                the origin ``[0, 0, 0]`` is used.
            quadrupol_tol : float, optional
                Tolerance for the ratio of residual quadrupole terms relative to
                the dipole term.  Wavelengths where the quadrupole contribution
                exceeds this ratio raise a warning (default is ``0.15``).
            verbose : bool, default True
                Print progress information during GPM extraction.

            Returns
            -------
            None
                Initializes the structure in-place.

            Raises
            ------
            ValueError
                If the supplied parameters are inconsistent or insufficient for
                the Mie calculation.

            Notes
            -----
            * The electric and magnetic dipole Mie coefficients are converted to volume‑scaled
            polarizabilities following García‑Etxarri *et al.*, Opt. Express **19**,
            4815 (2011).
            * The torch device for all output tensors is the
              ``mie_particle`` device.
            """
            # prep and imports
            from torchgdm.tools.misc import to_np
            from torchgdm.tools.misc import get_default_device

            # --- preparation, tensor conversion
            assert type(mie_particle) == pmd.Particle, "Requires pymiediff particle"
            self.device = mie_particle.device

            if verbose:
                t0 = time.time()
                print("Extracting eff. dipole model from pymiediff...")

            # tensor conversion
            wavelengths = torch.as_tensor(
                wavelengths, dtype=DTYPE_FLOAT, device=self.device
            )
            wavelengths = torch.atleast_1d(wavelengths)
            k0 = 2 * torch.pi / wavelengths

            # mie coefficients
            mie_results = mie_ab_sphere_3d_AD(
                mie_particle=mie_particle,
                wavelengths=wavelengths,
                n_env=n_env,
                as_dict=True,
                backend=backend,
            )
            a_n = mie_results["a_n"]
            b_n = mie_results["b_n"]
            env = mie_results["environment"]
            n_env = mie_results["n_env"]
            r_enclosing = mie_results["r_enclosing"]

            # check if dipole approximation is good
            a_quadrupol_res = a_n[:, 1].abs()
            wls_violation_a = to_np(
                wavelengths[a_quadrupol_res.to("cpu") > quadrupol_tol]
            )
            if len(wls_violation_a) > 0:
                warnings.warn(
                    "Mie series: {} wavelengths with ".format(len(wls_violation_a))
                    + "significant residual electric quadrupole contribution: "
                    + "{} nm".format([round(r, 1) for r in wls_violation_a])
                )

            b_quadrupol_res = b_n[:, 1].abs()
            wls_violation_b = to_np(
                wavelengths[b_quadrupol_res.to("cpu") > quadrupol_tol]
            )
            if len(wls_violation_b) > 0:
                warnings.warn(
                    "Mie series: {} wavelengths with ".format(len(wls_violation_b))
                    + "significant residual magnetic quadrupole contribution: "
                    + "{} nm".format([round(r, 1) for r in wls_violation_b])
                )

            # convert to polarizabilities (units of volume)
            # see: García-Etxarri, A. et al. Optics Express 19, 4815 (2011)
            a_pE = 1j * 3 / 2 * a_n[:, 0] / k0**3 / n_env**1
            a_mH = 1j * 3 / 2 * b_n[:, 0] / k0**3 / n_env**3

            # populate 6x6 polarizabilities for all wavelengths
            alpha_6x6 = torch.zeros(
                (len(wavelengths), 6, 6), dtype=DTYPE_COMPLEX, device=self.device
            )
            alpha_6x6[:, torch.arange(3), torch.arange(3)] += a_pE.unsqueeze(1)
            alpha_6x6[:, torch.arange(3, 6), torch.arange(3, 6)] += a_mH.unsqueeze(1)

            # set center of mass
            if r0 is None:
                r0 = torch.as_tensor(
                    [0.0, 0.0, 0.0], dtype=DTYPE_FLOAT, device=self.device
                )
            else:
                r0 = torch.as_tensor(r0, dtype=DTYPE_FLOAT, device=self.device)
                r0 = r0.squeeze()
                assert len(r0) == 3

            # wrap up in a dictionary compatible with the point dipole structure class
            alpha_dict = dict(
                r0=r0,
                r0_MD=r0,
                r0_ED=r0,
                alpha_6x6=alpha_6x6,
                wavelengths=wavelengths,
                enclosing_radius=r_enclosing,
                k0_spectrum=k0,
                environment=env,
            )

            # - point polarizability structure with Mie dipolar response
            super().__init__(
                positions=r0,
                alpha_dicts=[alpha_dict],
                device=self.device,
            )

            if verbose:
                print("Done in {:.2}s.".format(time.time() - t0))

    class StructAutodiffMieGPM3D(tg.struct.StructGPM3D):
        """Autodiff-capable TorchGDM GPM structure from a `pymiediff.Particle`."""

        __name__ = "Autodiff-Mie based GPM (3D) structure class"

        def __init__(
            self,
            mie_particle,
            wavelengths: torch.Tensor,
            r_gpm: torch.Tensor,
            r_probe: torch.Tensor = None,
            n_src_pw_angles: int = 12,
            r_probe_add: float = 20,  # nm
            n_env: float = None,
            n_max: int = None,
            backend=None,
            r0: torch.Tensor = None,
            device: torch.device = None,
            verbose=True,
            progress_bar=True,
            **kwargs,
        ):
            """Build a GPM structure from a single ``pymiediff.Particle``.

            Parameters
            ----------
            mie_particle : pymiediff.Particle
                Particle model used for field generation.
            wavelengths : torch.Tensor
                Wavelengths in nm.
            r_gpm : torch.Tensor
                GPM support coordinates.
            r_probe : torch.Tensor, optional
                Probe coordinates for extraction.
            n_src_pw_angles : int, default=12
                Number of plane-wave incidence angles.
            r_probe_add : float, default=20
                Offset for autogenerated probe sphere.
            n_env : float, optional
                Environment refractive index override.
            n_max : int, optional
                Mie truncation order.
            backend : str, optional
                Backend override.
            r0 : torch.Tensor, optional
                Center position of the structure.
            device : torch.device, optional
                Target device.
            verbose : bool, default=True
                Print progress messages.
            progress_bar : bool, default=True
                Enable progress bar.
            **kwargs
                Forwarded to GPM extraction backend.
            """
            from torchgdm.tools.misc import get_default_device
            from torchgdm.struct.eff_model_tools import extract_gpm_from_tmatrix

            # --- preparation, tensor conversion
            assert type(mie_particle) == pmd.Particle, "Requires pymiediff particle"
            if device is not None:
                mie_particle.set_device(device)
            device = mie_particle.device
            self.device = device

            gpm_dict = extract_GPM_sphere_miediff(
                mie_particle=mie_particle,
                wavelengths=wavelengths,
                r_gpm=r_gpm,
                r_probe=r_probe,
                n_src_pw_angles=n_src_pw_angles,
                r_probe_add=r_probe_add,
                n_env=n_env,
                n_max=n_max,
                backend=backend,
                verbose=verbose,
                progress_bar=progress_bar,
                **kwargs,
            )

            # set center of mass
            if r0 is None:
                r0 = torch.as_tensor(
                    [0.0, 0.0, 0.0], dtype=DTYPE_FLOAT, device=self.device
                )
            else:
                r0 = torch.as_tensor(r0, dtype=DTYPE_FLOAT, device=self.device)
                r0 = r0.squeeze()
                assert len(r0) == 3

            super().__init__(positions=r0, gpm_dicts=[gpm_dict], device=self.device)


# %%
# test setup
# ----------
if __name__ == "__main__":

    # - config
    # wl0 = torch.linspace(500, 1000, 5)
    import time
    import numpy as np
    import matplotlib.pyplot as plt

    wl0 = torch.as_tensor([600, 700])
    k0 = 2 * torch.pi / wl0

    r_core = 150.0
    r_shell = r_core + 100.0
    n_core = 4.0
    n_shell = 3.0
    n_env = 1.0

    mat_core = pmd.materials.MatConstant(n_core**2)
    mat_shell = pmd.materials.MatConstant(n_shell**2)

    # - setup the particle
    part = pmd.Particle(
        mat_env=n_env,
        r_core=r_core,
        mat_core=mat_core,
        r_shell=r_shell,
        mat_shell=mat_shell,
    )
    print(part)

    # structMieGPM = StructAutodiffMieEffPola3D(part, wavelengths=wl0)
    structMieGPM = StructAutodiffMieGPM3D(part, r_gpm=36, wavelengths=wl0)

    # - illumination field(s)
    e_inc_list = [tg.env.freespace_3d.PlaneWave(e0p=1, e0s=0, inc_angle=0)]

    env = tg.env.freespace_3d.EnvHomogeneous3D(env_material=n_env**2)
    sim = tg.simulation.Simulation(
        structures=[structMieGPM],
        environment=env,
        illumination_fields=e_inc_list,
        wavelengths=wl0,
    )
    sim.run()
    cs = sim.get_spectra_crosssections()
    cs_mie = part.get_cross_sections(k0=k0, backend="torch")
    print(cs_mie["cs_ext"])
    print(cs["ecs"])
    print(cs_mie["cs_ext"] / cs["ecs"])

    projection = "xz"
    idx_wl = 0

    r_probe = tg.tools.geometry.coordinate_map_2d_square(
        2000, 51, r3=500, projection=projection
    )["r_probe"]
    nf_sim = sim.get_nearfield(wl0[idx_wl], r_probe=r_probe)
    nf_mie = part.get_nearfields(k0=k0[idx_wl], r_probe=r_probe, backend="torch")

    # multiple particles + AD
    r_core_ad = torch.tensor(r_core, dtype=DTYPE_FLOAT)
    r_core_ad.requires_grad = True

    p_ad = pmd.Particle(
        mat_env=n_env,
        r_core=r_core_ad,
        mat_core=mat_core,
        r_shell=r_shell,
        mat_shell=mat_shell,
    )

    gpm_dict = extract_GPM_sphere_miediff(p_ad, wavelengths=wl0, r_gpm=36)

    pos_multi = tg.tools.geometry.coordinate_map_1d_circular(r=800, n_phi=6)["r_probe"]
    struct_multi = tg.struct3d.StructGPM3D(pos_multi, len(pos_multi) * [gpm_dict])

    sim_multi = tg.simulation.Simulation(
        structures=struct_multi,
        environment=env,
        illumination_fields=e_inc_list,
        wavelengths=wl0,
    )
    sim_multi.run()

    nf_sim_multi = sim_multi.get_nearfield(wl0[idx_wl], r_probe=r_probe)

    cs_multi = sim_multi.get_spectra_crosssections()
    cs_multi["ecs"][0].backward()
    print(cs_multi["ecs"][0])
    print(r_core_ad.grad)

    #
    # T-matrix multi-particle
    # -----------------
    import treams

    t0 = time.time()

    k0_treams = k0[idx_wl].cpu().numpy()

    materials = [
        treams.Material(n_core**2),
        treams.Material(n_shell**2),
        treams.Material(n_env**2),
    ]
    lmax = 5
    spheres = [
        treams.TMatrix.sphere(lmax, k0_treams, [r_core, r_shell], materials)
        for i in range(len(pos_multi))
    ]
    tm = treams.TMatrix.cluster(spheres, pos_multi.cpu().numpy()).interaction.solve()

    e0_pol = [1, 0, 0]
    inc = treams.plane_wave(
        [0, 0, k0_treams], pol=e0_pol, k0=tm.k0, material=tm.material
    )
    sca = tm @ inc.expand(tm.basis)

    # calc nearfield
    grid = r_probe.cpu().numpy()
    e_sca_treams = sca.efield(grid)
    e_inc_treams = inc.efield(grid)
    e_tot_treams = e_sca_treams + e_inc_treams
    intensity = np.sum(np.abs(e_sca_treams) ** 2, -1)

    print("total time treams: {:.2f}s".format(time.time() - t0))

    #

    plt.figure(figsize=(10, 7))

    # --- single sphere
    plt.subplot(221, title="GPM")
    im = tg.visu2d.scalarfield(
        nf_sim["sca"].get_efield_intensity()[0],
        positions=r_probe,
        projection=projection,
    )
    plt.colorbar(im)
    clim = im.get_clim()
    plt.subplot(222, title="Mie")
    im = tg.visu2d.scalarfield(
        (torch.abs(nf_mie["E_s"]) ** 2).sum(-1),
        positions=r_probe,
        projection=projection,
    )
    plt.colorbar(im)
    im.set_clim(clim)

    # --- multiple spheres
    plt.subplot(223, title="multi-GPM")
    im = tg.visu2d.scalarfield(
        nf_sim_multi["sca"].get_efield_intensity()[0],
        positions=r_probe,
        projection=projection,
    )
    sim_multi.plot_contour(projection=projection)
    plt.colorbar(im)
    clim = im.get_clim()

    plt.subplot(224, title="multi-treams")
    im = tg.visu2d.scalarfield(
        intensity,
        positions=r_probe,
        projection=projection,
    )
    sim_multi.plot_contour(projection=projection)
    plt.colorbar(im)
    im.set_clim(clim)

    plt.tight_layout()
    plt.show()
