# -*- coding: utf-8 -*-
"""
pymiediff.main
==============

High‑level interface for a single spherical (core‑shell) particle.

The module defines the :class:`Particle` class, which bundles core radius,
shell radius (optional), core/shell/environment materials and the device on
which calculations are performed.  It provides convenient methods to obtain
Mie coefficients, far‑field cross sections, angular scattering patterns and
near‑field values, automatically handling unit conversion, material
permittivities and Mie‑series truncation.

Typical usage
-------------

>>> import torch, pymiediff as pmd
>>> wl = torch.linspace(500, 1000, 100)
>>> k0 = 2 * torch.pi / wl
>>> p = Particle(
...     r_layers=torch.tensor([70.0, 100.0]),
...     eps_layers=torch.tensor([3.5**2, 4.0**2], dtype=torch.complex128),
...     mat_env=1.0,
... )
>>> cs = p.get_cross_sections(k0)   # dict with spectra (wavelength, q_ext, …)

The class is deliberately lightweight: it supports only a single particle.
For vectorised calculations over many particles see
``pymiediff.coreshell.cross_sections`` and related functions.

"""

import torch


class Particle:
    def __init__(
        self,
        r_layers=None,
        eps_layers=None,
        mat_env=1.0,
        device=None,
        r_core=None,
        mat_core=None,
        r_shell=None,
        mat_shell=None,
    ):
        """
        Initialise a single spherical particle (core‑only or core‑shell).

        Parameters
        ----------
        r_layers : torch.Tensor/array-like, optional
            Layer outer radii (nm), ordered from inner to outermost layer.
            This is the preferred multilayer input.

        eps_layers : torch.Tensor/array-like, optional
            Layer permittivities corresponding to ``r_layers``. Supported
            shapes are those accepted by ``pymiediff.coreshell`` functions
            (e.g. ``(L,)``, ``(L, N_k0)``, ``(N_part, L, N_k0)`` for batched use).
            This is the preferred multilayer input.

        r_core : float or torch.Tensor, optional
            Legacy core radius (nm), used when multilayer inputs are not given.

        mat_core : pymiediff.materials.Material or float/int/complex/torch.Tensor, optional
            Legacy core material. If a scalar is supplied, a constant-index
            material :class:`pymiediff.materials.MatConstant` is created from
            the value (interpreted as refractive index).

        r_shell : float or torch.Tensor, optional
            Legacy shell radius (nm). Must be supplied together with
            ``mat_shell``.

        mat_shell : pymiediff.materials.Material or float/int/complex/torch.Tensor, optional
            Legacy shell material.

        mat_env : pymiediff.materials.Material or float/int/complex/torch.Tensor, optional
            Surrounding (environment) material. Defaults to a refractive index of
            ``1.0`` (air). Scalars are converted to a constant‑index material.

        device : str or torch.device, optional
            Torch device on which all tensors will be allocated. If omitted,
            defaults to ``'cpu'``.

        Notes
        -----
        * Preferred mode: provide ``r_layers`` + ``eps_layers``.
        * Legacy mode: provide ``r_core`` + ``mat_core`` (and optionally
          ``r_shell`` + ``mat_shell``).
        * The constructor validates that both ``r_shell`` and ``mat_shell`` are
          either provided together or omitted together in legacy mode.
        * All radii and material parameters are internally stored as
          ``torch.Tensor`` objects on the specified ``device``.
        * Materials given as scalars are automatically wrapped in
          :class:`pymiediff.materials.MatConstant` with the square of the value
          (i.e. converting a refractive‑index ``n`` to permittivity ``ε = n²``).

        Raises
        ------
        AssertionError
            If inconsistent argument combinations are supplied.
        """
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        self.r_layers = None
        self.eps_layers = None
        self._use_layers = (r_layers is not None) or (eps_layers is not None)

        if (r_layers is None) ^ (eps_layers is None):
            raise ValueError("`r_layers` and `eps_layers` must be provided together.")

        if self._use_layers:
            if any(v is not None for v in (r_core, mat_core, r_shell, mat_shell)):
                raise ValueError(
                    "Use either multilayer inputs (`r_layers`, `eps_layers`) "
                    "or legacy core/shell inputs (`r_core`, `mat_core`, `r_shell`, `mat_shell`)."
                )
            self.r_layers = torch.as_tensor(r_layers, device=self.device)
            if self.r_layers.ndim != 1:
                raise ValueError("For `Particle`, `r_layers` must be one-dimensional (L,).")
            if self.r_layers.numel() < 1:
                raise ValueError("`r_layers` must contain at least one layer.")
            self.eps_layers = torch.as_tensor(eps_layers, device=self.device)
            self.r_c = self.r_layers[0]
            self.r_s = self.r_layers[-1]
            self.mat_c = None
            self.mat_s = None
        else:
            if r_core is None or mat_core is None:
                raise ValueError(
                    "Provide either (`r_layers`, `eps_layers`) or at least legacy "
                    "(`r_core`, `mat_core`)."
                )
            if (r_shell is None) ^ (mat_shell is None):
                raise ValueError(
                    "Either both, or none of shell radius and shell material must be given."
                )

            self.r_c = torch.as_tensor(r_core, device=self.device)  # core radius, nm
            if r_shell is not None:
                self.r_s = torch.as_tensor(r_shell, device=self.device)  # shell radius, nm
            else:
                self.r_s = None

            # create actual materials if float or int is given
            from pymiediff.materials import MatConstant

            if type(mat_core) in (float, int, complex, torch.Tensor):
                self.mat_c = MatConstant(mat_core**2, device=self.device)
            else:
                self.mat_c = mat_core
                self.mat_c.set_device(self.device)

            if mat_shell is not None:
                if type(mat_shell) in (float, int, complex, torch.Tensor):
                    self.mat_s = MatConstant(mat_shell**2, device=self.device)
                else:
                    self.mat_s = mat_shell
                    self.mat_s.set_device(self.device)
            else:
                self.mat_s = None

        from pymiediff.materials import MatConstant
        if type(mat_env) in (float, int, complex, torch.Tensor):
            self.mat_env = MatConstant(mat_env**2, device=self.device)
        else:
            self.mat_env = mat_env
            self.mat_env.set_device(self.device)

    def set_device(self, device):
        self.device = device

        self.r_c = self.r_c.to(device=self.device)
        if self.r_s is not None:
            self.r_s = self.r_s.to(device=self.device)
        if self.r_layers is not None:
            self.r_layers = self.r_layers.to(device=self.device)
        if self.eps_layers is not None:
            self.eps_layers = self.eps_layers.to(device=self.device)

        if self.mat_c is not None:
            self.mat_c.set_device(self.device)
        if self.mat_s is not None:
            self.mat_s.set_device(self.device)
        self.mat_env.set_device(self.device)

    def __repr__(self):
        out_str = ""
        if self._use_layers:
            out_str += "multilayer particle (on device: {})\n".format(self.device)
            out_str += " - layers   = {}\n".format(int(self.r_layers.numel()))
            out_str += " - radii    = {}nm\n".format(self.r_layers.data)
        elif self.r_s is None:
            out_str += "homogeneous particle (on device: {})\n".format(self.device)
            out_str += " - radius   = {}nm\n".format(self.r_c.data)
            out_str += " - material : {}\n".format(self.mat_c.__name__)
        else:
            out_str += "core-shell particle\n"
            out_str += " - core radius    = {}nm\n".format(self.r_c.data)
            out_str += " - shell radius   = {}nm\n".format(self.r_s.data)
            out_str += " - core material  : {}\n".format(self.mat_c.__name__)
            out_str += " - shell material : {}\n".format(self.mat_s.__name__)
        out_str += " - environment    : {}\n".format(self.mat_env.__name__)
        return out_str

    def get_material_permittivities(self, k0: torch.Tensor) -> tuple:
        """
        Return spectral permittivities of core, shell and environment.

        Parameters
        ----------
        k0 : torch.Tensor
            Tensor containing all evaluation wavenumbers (rad nm^-1).

        Returns
        -------
        tuple of torch.Tensor
            (eps_c, eps_s, eps_env)
                eps_c : core permittivity evaluated at ``k0``.
                eps_s : shell permittivity evaluated at ``k0`` (or equal to ``eps_c`` for a homogeneous particle).
                eps_env : environment permittivity evaluated at ``k0``.
        """
        if self._use_layers:
            raise RuntimeError(
                "`get_material_permittivities` is only available in legacy core/shell mode."
            )
        k0 = torch.as_tensor(k0, device=self.device)
        wl0 = 2 * torch.pi / k0

        eps_c = self.mat_c.get_epsilon(wavelength=wl0)
        eps_env = self.mat_env.get_epsilon(wavelength=wl0)

        if self.mat_s is None:
            r_s = self.r_c
            eps_s = eps_c
        else:
            r_s = self.r_s
            eps_s = self.mat_s.get_epsilon(wavelength=wl0)

        return eps_c, eps_s, eps_env

    def get_mie_coefficients(
        self, k0: torch.Tensor, return_internal=False, **kwargs
    ) -> dict:
        """
        Compute Mie coefficients for the particle.

        Parameters
        ----------
        k0 : torch.Tensor
            Evaluation wavenumbers (rad nm^-1).  The tensor is moved to the
            particle's device internally.
        return_internal : bool, optional
            If ``True`` also return the internal Mie coefficients
            (``c_n``, ``d_n``, ``f_n``, ``g_n``, ``v_n``, ``w_n``).  Default is
            ``False``.
        **kwargs : dict
            Additional keyword arguments passed to
            :func:`pymiediff.coreshell.mie_coefficients`.  Typical options
            include ``n_max`` to manually set the truncation order.

        Returns
        -------
        dict
            Dictionary containing the external Mie coefficients and related
            parameters.  Keys include:

            - ``a_n`` : external electric Mie coefficient
            - ``b_n`` : external magnetic Mie coefficient
            - ``k0``  : evaluation wavenumbers
            - ``k``   : wavenumbers in the host medium
            - ``n``   : Mie orders
            - ``n_max`` : maximum Mie order used
            - ``r_c`` : core radius
            - ``r_s`` : shell radius (or core radius for homogeneous particles)
            - ``eps_c`` : core permittivity spectrum
            - ``eps_s`` : shell permittivity spectrum
            - ``eps_env`` : environmental permittivity spectrum
            - ``n_c`` : core refractive index
            - ``n_s`` : shell refractive index
            - ``n_env`` : environmental refractive index

            If ``return_internal`` is ``True``, the dictionary also contains:

            - ``c_n`` : internal magnetic Mie coefficient (core)
            - ``d_n`` : internal electric Mie coefficient (core)
            - ``f_n`` : internal magnetic Mie coefficient - first kind (shell)
            - ``g_n`` : internal electric Mie coefficient - first kind (shell)
            - ``v_n`` : internal magnetic Mie coefficient - second kind (shell)
            - ``w_n`` : internal electric Mie coefficient - second kind (shell)

        Notes
        -----
        The Mie series truncation follows the Wiscombe criterion
        (Wiscombe, *Appl. Opt.* **19**, 1505‑1509 (1980)).  The helper
        function ``_squeeze_dimensions`` removes singleton dimensions for a
        single‑particle calculation.

        """
        from pymiediff.coreshell import mie_coefficients

        k0 = torch.as_tensor(k0, device=self.device)
        wl0 = 2 * torch.pi / k0
        eps_env = self.mat_env.get_epsilon(wavelength=wl0)
        if self._use_layers:
            res = mie_coefficients(
                k0,
                r_layers=self.r_layers,
                eps_layers=self.eps_layers,
                eps_env=eps_env,
                **kwargs,
            )
        else:
            eps_c, eps_s, _ = self.get_material_permittivities(k0)
            r_s = self.r_c if (self.r_s is None) else self.r_s
            res = mie_coefficients(
                k0,
                r_c=self.r_c,
                r_s=r_s,
                eps_c=eps_c,
                eps_s=eps_s,
                eps_env=eps_env,
                **kwargs,
            )

        # single particle: remove empty dimension
        from pymiediff.helper.helper import _squeeze_dimensions

        _squeeze_dimensions(res)  # in place

        return res

    def get_cross_sections(self, k0: torch.Tensor, **kwargs) -> dict:
        """
        Compute far‑field cross‑section spectra.

        Parameters
        ----------
        k0 : torch.Tensor
            Tensor of evaluation wavenumbers (rad nm^-1).  Will be cast to the
            particle's device automatically.
        **kwargs :
            Additional keyword arguments passed to
            :func:`pymiediff.coreshell.cross_sections`.

        Returns
        -------
        dict
            Dictionary containing the spectral results.  Keys include:

            - ``wavelength`` : torch.Tensor
                Wavelengths (nm) corresponding to the spectra.
            - ``q_ext`` : torch.Tensor
                Extinction efficiency.
            - ``q_sca`` : torch.Tensor
                Scattering efficiency.
            - ``q_abs`` : torch.Tensor
                Absorption efficiency.
            - ``c_ext`` : torch.Tensor
                Extinction cross‑section (nm^2).
            - ``c_sca`` : torch.Tensor
                Scattering cross‑section (nm^2).
            - ``c_abs`` : torch.Tensor
                Absorption cross‑section (nm^2).

            Any additional fields returned by
            :func:`pymiediff.coreshell.cross_sections` are also included.

        Notes
        -----
        The Wiscombe criterion is used internally to truncate the Mie series.
        The result is squeezed to remove the singleton particle dimension.
        """
        from pymiediff.coreshell import cross_sections

        k0 = torch.as_tensor(k0, device=self.device)
        wl0 = 2 * torch.pi / k0
        eps_env = self.mat_env.get_epsilon(wavelength=wl0)
        if self._use_layers:
            res = cross_sections(
                k0,
                r_layers=self.r_layers,
                eps_layers=self.eps_layers,
                eps_env=eps_env,
                **kwargs,
            )
        else:
            eps_c, eps_s, _ = self.get_material_permittivities(k0)
            r_s = self.r_c if (self.r_s is None) else self.r_s
            res = cross_sections(
                k0,
                r_c=self.r_c,
                r_s=r_s,
                eps_c=eps_c,
                eps_s=eps_s,
                eps_env=eps_env,
                **kwargs,
            )

        # single particle: remove empty dimension
        from pymiediff.helper.helper import _squeeze_dimensions

        res = _squeeze_dimensions(res)

        return res

    def get_angular_scattering(
        self, k0: torch.Tensor, theta: torch.Tensor, **kwargs
    ) -> dict:
        """
        Compute angular scattering for a single particle.

        Parameters
        ----------
        k0 : torch.Tensor
            Evaluation wavenumbers (rad nm^-1).  Will be moved to the particle's
            device internally.
        theta : torch.Tensor
            Scattering angles (rad).  Can be any shape that broadcasts with
            ``k0``.
        **kwargs : dict
            Additional keyword arguments passed to
            :func:`pymiediff.coreshell.angular_scattering`.  Typical options
            include ``n_max`` to manually set the truncation order.

        Returns
        -------
        dict
            Dictionary containing angular‑scattering results.  Keys include
            (but are not limited to):

            - ``theta`` : torch.Tensor
                The input angles (rad) after possible broadcasting.
            - ``i_unpol`` : torch.Tensor
                Unpolarised intensity as a function of ``theta`` and ``k0``.
            - ``i_par`` : torch.Tensor
                Parallel‑polarised intensity.
            - ``i_perp`` : torch.Tensor
                Perpendicular‑polarised intensity.

            Any additional fields returned by
            :func:`pymiediff.coreshell.angular_scattering` are also present.

        Notes
        -----
        The helper function ``_squeeze_dimensions`` is applied to the result
        to remove the singleton particle dimension for single‑particle
        calculations.
        """
        from pymiediff.coreshell import angular_scattering

        k0 = torch.as_tensor(k0, device=self.device)
        theta = torch.as_tensor(theta, device=self.device)

        wl0 = 2 * torch.pi / k0
        eps_env = self.mat_env.get_epsilon(wavelength=wl0)
        if self._use_layers:
            res_angSca = angular_scattering(
                k0=k0,
                theta=theta,
                r_layers=self.r_layers,
                eps_layers=self.eps_layers,
                eps_env=eps_env,
                **kwargs,
            )
        else:
            eps_c, eps_s, _ = self.get_material_permittivities(k0)
            r_s = self.r_c if (self.r_s is None) else self.r_s
            res_angSca = angular_scattering(
                k0=k0,
                theta=theta,
                r_c=self.r_c,
                r_s=r_s,
                eps_c=eps_c,
                eps_s=eps_s,
                eps_env=eps_env,
                **kwargs,
            )

        # single particle: remove empty dimension
        from pymiediff.helper.helper import _squeeze_dimensions

        res_angSca = _squeeze_dimensions(res_angSca)

        return res_angSca

    def get_nearfields(self, k0: torch.Tensor, r_probe: torch.Tensor, **kwargs) -> dict:
        """
        Compute electric and magnetic near-fields at probe positions

        Parameters
        ----------
        k0 : torch.Tensor
            Evaluation wavenumbers (rad nm^-1).  Will be cast to the particle's
            device automatically.
        r_probe : torch.Tensor
            Cartesian probe positions with shape ``(..., 3)`` where the last
            dimension indexes the ``x, y, z`` coordinates.
        **kwargs : dict
            Additional keyword arguments passed to
            :func:`pymiediff.coreshell.nearfields`.  The illumination amplitude is
            fixed to ``E_0 = 1``.

        Returns
        -------
        dict
            Dictionary containing the fields:

            - ``E_s`` : torch.Tensor
                Scattered electric field at each probe position.
            - ``H_s`` : torch.Tensor
                Scattered magnetic field at each probe position.
            - ``E_t`` : torch.Tensor
                Total electric field at each probe position (scat+inc).
            - ``H_t`` : torch.Tensor
                Total magnetic field at each probe position (scat+inc).
            - ``E_i`` : torch.Tensor
                Incident electric field at each probe position.
            - ``H_i`` : torch.Tensor
                Incident magnetic field at each probe position.

            Any extra entries returned by ``nearfields`` are also included.

        Notes
        -----
            The method internally calls :func:`pymiediff.coreshell.nearfields`,
            handling material permittivities and radius selection.  For a
            single-particle calculation the singleton particle dimension is
            squeezed from the output.
        """
        from pymiediff.coreshell import nearfields

        k0 = torch.as_tensor(k0, device=self.device)
        r_probe = torch.as_tensor(r_probe, device=self.device)
        assert r_probe.shape[-1] == 3

        wl0 = 2 * torch.pi / k0
        eps_env = self.mat_env.get_epsilon(wavelength=wl0)
        if self._use_layers:
            res_nf = nearfields(
                k0=k0,
                r_probe=r_probe,
                r_layers=self.r_layers,
                eps_layers=self.eps_layers,
                eps_env=eps_env,
                **kwargs,
            )
        else:
            eps_c, eps_s, _ = self.get_material_permittivities(k0)
            r_s = self.r_c if (self.r_s is None) else self.r_s
            res_nf = nearfields(
                k0=k0,
                r_probe=r_probe,
                r_c=self.r_c,
                r_s=r_s,
                eps_c=eps_c,
                eps_s=eps_s,
                eps_env=eps_env,
                **kwargs,
            )

        # single particle: remove empty dimension
        from pymiediff.helper.helper import _squeeze_dimensions

        res_nf = _squeeze_dimensions(res_nf)

        return res_nf


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    import pymiediff as pmd

    # - config
    wl0 = torch.linspace(500, 1000, 50)
    k0 = 2 * torch.pi / wl0

    r_core = 70.0
    r_shell = 100.0
    mat_core = pmd.materials.MatDatabase("Si")
    mat_shell = pmd.materials.MatDatabase("Ge")
    n_env = 1.0

    # - setup the particle
    p = Particle(
        mat_env=n_env,
        r_core=r_core,
        mat_core=mat_core,
        r_shell=r_shell,
        mat_shell=mat_shell,
    )
    print(p)

    # - efficiency spectra
    cs = p.get_cross_sections(k0)
    plt.figure()
    plt.plot(cs["wavelength"], cs["q_ext"], label="$Q_{ext}$")
    plt.plot(cs["wavelength"], cs["q_sca"], label="$Q_{sca}$")
    plt.plot(cs["wavelength"], cs["q_abs"], label="$Q_{abs}$")
    plt.xlabel("wavelength (nm)")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # - scattering radiation pattern
    theta = torch.linspace(0.0, 2 * torch.pi, 100)
    angular = p.get_angular_scattering(k0, theta)

    plt.figure(figsize=(12, 2))
    for i, i_k0 in enumerate(range(len(k0))[::5]):
        ax = plt.subplot(1, 10, i + 1, polar=True)
        plt.title(f"{wl0[i_k0]:.1f} nm")
        ax.plot(angular["theta"], angular["i_unpol"][i_k0])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout()
    plt.show()
