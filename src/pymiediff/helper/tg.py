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

import torch
import pymiediff as pmd
import warnings

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


# ------ eff. dipole pair extraction for core-shell sphere via pymiediff -------
def mie_ab_sphere_3d_AD(
    mie_particle,
    wavelengths: torch.Tensor,
    n_env=None,
    n_max=2,
    as_dict=False,
):
    """
    Compute 3‑D Mie scattering coefficients for a (core‑shell) sphere.

    Parameters
    ----------
    mie_particle : :class:`pymiediff.Particle`
        The Mie particle to build a GPM model for.
    wavelengths : torch.Tensor
        Wavelengths (nm) at which to evaluate the coefficients.
    n_env : float, optional
        Refractive index of the surrounding medium.
        If not provided, use the environment defined in the particle.
    n_max : int, default 2
        Maximum Mie multipole order (dipole‑only if ``2``).
    as_dict : bool, default False
        If ``True`` return a dictionary with detailed results, otherwise
        return ``(a_n, b_n)`` only.

    Returns
    -------
    tuple or dict
        ``(a_n, b_n)`` when ``as_dict=False``; otherwise a dict containing
        ``a_n``, ``b_n``, ``environment``, ``n_env``, ``device``,
        ``r_enclosing`` and ``wavelengths``.

    Notes
    -----
    * The torch device for all output tensors is the `mie_particle` device.
    """
    from torchgdm.env import EnvHomogeneous3D

    # --- preparation, tensor conversion
    assert type(mie_particle) == pmd.Particle, "Requires pymiediff particle"
    device = mie_particle.device

    wavelengths = torch.as_tensor(wavelengths, dtype=DTYPE_FLOAT, device=device)
    wavelengths = torch.atleast_1d(wavelengths)
    k0 = 2 * torch.pi / wavelengths

    eps_c, eps_s, eps_env = mie_particle.get_material_permittivities(k0)
    eps_env = torch.atleast_1d(eps_env)
    assert torch.all(eps_env == eps_env[0]), "dispersive environment not supported yet"
    if n_env is not None:
        assert eps_env[0] == n_env**2
        eps_env = n_env**2
    env_3d = EnvHomogeneous3D(env_material=float(eps_env[0].real), device=device)

    # --- get Mie coefficients from pymiediff particle
    miecoeff = mie_particle.get_mie_coefficients(k0=k0, n_max=n_max)

    a_n = miecoeff["a_n"]
    b_n = miecoeff["b_n"]

    if a_n.ndim == 1:  # single wl
        a_n = a_n.unsqueeze(1)  # add empty wavelength dimension
    if b_n.ndim == 1:
        b_n = b_n.unsqueeze(1)

    a_n = a_n.moveaxis(0, 1)
    b_n = b_n.moveaxis(0, 1)  # move Mie-order last

    # full radius
    if mie_particle.r_s is None:
        r_enclosing = mie_particle.r_c  # homogeneous sphere radius
    else:
        r_enclosing = mie_particle.r_s  # outer radius

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


# ------------- GPM extraction for core-shell sphere via pymiediff -------------
def rotation_x(alpha, device="cpu", DTYPE_FLOAT=torch.float32):
    """matrix for clockwise rotation around x-axis by angle `alpha` (in radian)"""
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
    """matrix for clockwise rotation around y-axis by angle `alpha` (in radian)"""
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
    """matrix for clockwise rotation around z-axis by angle alpha (in radian)"""
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
    try:
        # ignore import warnings
        with warnings.catch_warnings():
            import treams
    except ModuleNotFoundError:
        print("Requires `treams`, install via `pip install treams`.")
        raise

    inc_fields_conf = []
    for angle in torch.linspace(0, 2 * torch.pi, n_angles + 1)[:-1]:
        for inc in inc_planes:
            inc_fields_conf.append(dict(angle=angle, inc_plane=inc, polarization="s"))
    return inc_fields_conf


# - parallelized treams evaluation (illumination and scattering)
def _eval_mie(mie_particle, inc_conf, k0, r_probe, r_gpm, n_max=None):
    """
    Compute near‑field electric and magnetic fields for a Mie particle under a
    single plane‑wave illumination.

    Parameters
    ----------
    mie_particle : :class:`pymiediff.Particle`
        The scattering particle for which the fields are evaluated.
    inc_conf : dict
        Configuration of the incident plane wave with keys:
        ``"polarization"`` (``"s"`` or ``"p"``), ``"inc_plane"`` (``"xy"``,
        ``"xz"``, or ``"yz"``) and ``"angle"`` (incidence angle in radians).
    k0 : float or torch.Tensor
        Vacuum wavenumber (2pi/wavelength). Can be a scalar or a 0-dim tensor.
    r_probe : torch.Tensor
        Probe positions where the scattered field is sampled, shape
        ``(N_probe, 3)``.
    r_gpm : torch.Tensor
        Positions of the GPM, shape
        ``(N_gpm, 3)``.
    n_max : int, optional
        Maximum multipole order for the Mie expansion. If ``None`` the
        particle decides internally.

    Returns
    -------
    tuple of torch.Tensor
        ``(e_sca, h_sca, e_inc, h_inc)`` where each tensor has shape
        ``(N, 3)`` with ``N`` being ``N_probe`` for the scattered fields and
        ``N_gpm`` for the incident fields, respectively.
    """
    # - calc incident and scattered fields
    pol_type = inc_conf["polarization"]
    inc_plane = inc_conf["inc_plane"]
    inc_angle = inc_conf["angle"]  # rad

    # rotate grid points
    if inc_plane == "xy":
        rot = rotation_z(-inc_angle, device=mie_particle.device)
    if inc_plane == "xz":
        rot = rotation_y(-inc_angle, device=mie_particle.device)
    if inc_plane == "yz":
        rot = rotation_x(-inc_angle, device=mie_particle.device)

    r_probe_rot = torch.matmul(r_probe, rot)
    r_gpm_rot = torch.matmul(r_gpm, rot)

    # polarization adjustment: exchange coordinates dependent on incidence plane
    if pol_type == "p":
        raise ValueError("polarization p not implemented yet")
        # r_probe = r_probe[..., [1, 0, 2]]

    # caclulate nearfields with pymiediff
    fields_sca = mie_particle.get_nearfields(k0, r_probe_rot, n_max=n_max)
    fields_inc = mie_particle.get_nearfields(k0, r_gpm_rot, n_max=n_max)

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
    verbose=True,
    progress_bar=True,
    **kwargs,
):
    """
    Extract a global polarizability matrix (GPM) model for a core-shell sphere

    This function evaluates the near‑fields for a core-shell sphere using *pymiediff*.
    It is therefore fully auto-differentiable.

    Parameters
    ----------
    mie_particle : :class:`pymiediff.Particle`
        The Mie particle to build a GPM model for.
    wavelengths : torch.Tensor
        Wavelength(s) (nm) at which the GPM is constructed.
    r_gpm : int or torch.Tensor
        If an ``int`` is given, a spherical sampling of ``r_gpm`` points
        (approximately ``sqrt(r_gpm)`` per polar/azimuthal direction) is
        generated on a sphere of radius ``r_inner = r_shell/3``.  If a tensor
        is supplied, it is used directly as the GPM probe positions.
    r_probe : torch.Tensor, optional
        Probe positions for the scattered near-field evaluation. If ``None``,
        a default spherical shell at ``r_shell + r_probe_add`` is used.
    n_src_pw_angles : int, default 12
        Number of incident plane‑wave directions per azimuthal plane.
    r_probe_add : float, default 20
        Radial offset (nm) for the default ``r_probe`` shell.
    n_env : float, optional
        Refractive index of the surrounding medium.
        If not provided, use the environment defined in the particle.
    n_max : int, optional
        Maximum multipole order for the internal ``pymiediff`` near‑field
        calculation. If ``None``, automatic cutoff will be used.
    verbose : bool, default True
        Print progress information during GPM extraction.
    progress_bar : bool, default True
        No effect so far. Intended to show a tqdm progress bar while processing wavelengths.
    **kwargs : dict
        Additional keyword arguments passed to
        :func:`torchgdm.struct.eff_model_tools.extract_gpm_from_fields`.

    Returns
    -------
    dict
        Dictionary containing the GPM data:
        ``r_gpm`` (probe positions),
        ``GPM_N6xN6`` (tensor of shape ``(N_wl, N_gpm, 6, 6)``),
        ``wavelengths``, ``full_geometry``, ``n_gpm_dp``,
        ``r0``, ``enclosing_radius``, ``k0_spectrum``,
        ``environment``, ``extraction_r_probe`` and the original
        ``mie_particle``.

    Notes
    -----
    * The function builds a set of incident plane‑wave configurations
      spanning the ``xz``, ``yz`` and ``xy`` incidence planes.
    * Near‑fields are obtained via :meth:`pymiediff.Particle.get_nearfields`.
    * The GPM is optimized with
      :func:`torchgdm.struct.eff_model_tools.extract_gpm_from_fields`,
      which returns a full set of results; only the ``GPM`` entry is kept.
    * The dummy geometry generated at the end is required by
      :class:`torchgdm.struct.StructGPM3D` but does not affect the GPM
      itself.
    * The torch device for all output tensors is the `mie_particle` device.
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

    eps_c, eps_s, eps_env = mie_particle.get_material_permittivities(k0)
    eps_env = torch.atleast_1d(eps_env)
    assert torch.all(eps_env == eps_env[0]), "dispersive environment not supported yet"
    if n_env is not None:
        assert eps_env[0] == n_env**2
        eps_env = n_env**2
    env_3d = EnvHomogeneous3D(env_material=float(eps_env[0].real), device=device)
    
    # full radius
    if mie_particle.r_s is None:
        r_enclosing = mie_particle.r_c  # homogeneous sphere radius
    else:
        r_enclosing = mie_particle.r_s  # outer radius


    # --- gpm locations and extraction probe points
    if verbose:
        t0 = time.time()
        print("Extracting GPM model via pymiediff near-field eval...")

    if r_gpm is None:
        r_gpm = DEFAULT_R_GPM

    if type(r_gpm) in (int, float):
        r_gpm = int(r_gpm)
        r_inner = r_enclosing / 3  # nm
        r_gpm = tg.tools.geometry.coordinate_map_2d_spherical(
            r=r_inner,
            n_phi=int(r_gpm**0.5),
            n_teta=int(r_gpm**0.5),
            device=mie_particle.device,
        )["r_probe"]

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
            _eval_mie(mie_particle, inc_conf, k0_single, r_probe, r_gpm, n_max=n_max)
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
            norm = 1.0

            e_sca_mie[i_inc] = fields[0] / norm
            h_sca_mie[i_inc] = fields[1] / norm
            e_inc_mie[i_inc] = fields[2] / norm
            h_inc_mie[i_inc] = fields[3] / norm

        # - optimize GPM
        gpm_dict = extract_gpm_from_fields(
            wavelength=2 * torch.pi / k0_single,
            efields_sca=torch.as_tensor(e_sca_mie, device=device).to(
                dtype=DTYPE_COMPLEX
            ),
            hfields_sca=torch.as_tensor(h_sca_mie, device=device).to(
                dtype=DTYPE_COMPLEX
            ),
            efields_inc=torch.as_tensor(e_inc_mie, device=device).to(
                dtype=DTYPE_COMPLEX
            ),
            hfields_inc=torch.as_tensor(h_inc_mie, device=device).to(
                dtype=DTYPE_COMPLEX
            ),
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
            r0: torch.Tensor = None,
            quadrupol_tol=0.15,
            verbose=True,
        ):
            """
            Initialize a 3‑D point polarizability for a core‑shell sphere using
            dipolar order Mie theory.

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
            * The torch device for all output tensors is the `mie_particle` device.
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
        """
        Autodiff‑enabled 3‑D Global Polarizability Matrix (GPM) structure based on
        Mie‑theory for a core‑shell sphere.

        This class builds a GPM model from a :class:`pymiediff.Particle` by
        automatically extracting near‑field data with *pymiediff* and fitting an
        effective description using multiple dipole pairs via :func:`torchgdm.struct.eff_model_tools.extract_gpm_from_fields`.

        The resulting structure can be used directly in a ``torchgdm`` simulation
        (e.g. as part of a ``Simulation``) and supports full autograd
        differentiation with respect to particle parameters (radii, refractive
        indices, etc.) because all intermediate quantities and operations are implemented in torch.

        Notes
        -----
        * The GPM extraction can be computationally intensive; consider using a
        modest ``n_src_pw_angles`` and a reduced ``r_gpm`` for quick tests with lower accuracy.
        * The class stores the full ``gpm_dict`` (including metadata) and forwards
        the extracted GPM to the parent ``StructGPM3D`` constructor.
        """

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
            r0: torch.Tensor = None,
            device: torch.device = None,
            verbose=True,
            progress_bar=True,
            **kwargs,
        ):
            """3D GPM structure from a Mie‑theory particle (autodiff capable)

            The constructor extracts near‑field data for the specified
            ``wavelengths``, builds an effective dipole-pair-based model, and
            initialises the underlying :class:`torchgdm.struct.StructGPM3D` with the
            resulting GPM dictionary.

            Parameters
            ----------
            mie_particle : :class:`pymiediff.Particle`
                Core‑shell particle for which the GPM is constructed.
            wavelengths : torch.Tensor
                Wavelengths (nm) at which the GPM is evaluated.
            r_gpm : int or torch.Tensor
                Number of GPM probe positions (int) or explicit probe coordinates.
                If an ``int`` is given, a spherical sampling of ``r_gpm`` points
                (approximately ``sqrt(r_gpm)`` per polar/azimuthal direction) is
                generated on a sphere of radius ``r_inner = r_shell/3``.
            r_probe : torch.Tensor, optional
                Probe positions for the scattered‑field evaluation; if ``None`` a
                default spherical shell is generated.
            n_src_pw_angles : int, default 12
                Number of incident plane‑wave directions per incidence plane.
            r_probe_add : float, default 20
                Radial offset (nm) for the default ``r_probe`` shell.
            n_env : float, optional
                Refractive index of the surrounding medium; defaults to the
                particle’s environment.
            n_max : int, optional
                Maximum multipole order for the internal near‑field calculation.
            r0 : torch.Tensor, optional
                Global translation of the structure (default origin).
            device : torch.device, optional
                Device on which tensors are allocated. If ``None``, the device of
                ``mie_particle`` is used.
            verbose : bool, default True
                Print progress information during GPM extraction.
            progress_bar : bool, default True
                No effect so far. Intended to show a tqdm progress bar while processing wavelengths.
            **kwargs
                Additional keyword arguments forwarded to the GPM extraction routine.

            Raises
            ------
            AssertionError
                If ``mie_particle`` is not an instance of :class:`pymiediff.Particle`.

            Notes
            -----
            The heavy‑lifting is performed by :func:`extract_GPM_sphere_miediff`,
            which returns a dictionary containing the GPM tensor and metadata.
            This dictionary is then passed to the parent ``StructGPM3D`` constructor.
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
        r_core=r_core,
        r_shell=r_shell,
        mat_core=mat_core,
        mat_shell=mat_shell,
        mat_env=n_env,
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
    cs_mie = part.get_cross_sections(k0=k0)
    print(cs_mie["cs_ext"])
    print(cs["ecs"])
    print(cs_mie["cs_ext"] / cs["ecs"])

    projection = "xz"
    idx_wl = 0

    r_probe = tg.tools.geometry.coordinate_map_2d_square(
        2000, 51, r3=500, projection=projection
    )["r_probe"]
    nf_sim = sim.get_nearfield(wl0[idx_wl], r_probe=r_probe)
    nf_mie = part.get_nearfields(k0=k0[idx_wl], r_probe=r_probe)

    # multiple particles + AD
    r_core_ad = torch.tensor(r_core, dtype=DTYPE_FLOAT)
    r_core_ad.requires_grad = True

    p_ad = pmd.Particle(
        r_core=r_core_ad,
        r_shell=r_shell,
        mat_core=mat_core,
        mat_shell=mat_shell,
        mat_env=n_env,
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
