# -*- coding: utf-8 -*-
"""
pymiediff.main
==============

High‑level interface for a single spherical (core / multishell) particle.

The module defines the :class:`Particle` class, which bundles core radius,
shell(s) radius (optional), core/shells/environment materials and the device on
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
``pymiediff.multishell.cross_sections`` and related functions.

Notes
-----

Both APIs are supported:

- multilayer-first inputs (``r_layers`` with ``eps_layers`` or ``mat_layers``),
- legacy core/shell inputs (``r_core``, ``mat_core``, optional shell).
"""

import torch


class Particle:
    """Spherical particle container used by high-level pymiediff workflows."""

    def __init__(
        self,
        r_layers=None,
        eps_layers=None,
        mat_layers=None,
        mat_env=1.0,
        device=None,
        r_core=None,
        mat_core=None,
        r_shell=None,
        mat_shell=None,
    ):
        """Create a spherical particle model.

        Parameters
        ----------
        r_layers : torch.Tensor/array-like, optional
            Layer outer radii (nm), ordered from inner to outermost layer.
            This is the preferred multilayer input.

        eps_layers : torch.Tensor/array-like, optional
            Layer permittivities corresponding to ``r_layers``. Supported
            shapes are those accepted by ``pymiediff.multishell`` functions
            (e.g. ``(L,)``, ``(L, N_k0)``, ``(N_part, L, N_k0)`` for batched use).
            This is the preferred multilayer input.

        mat_layers : list, optional
            Layer materials corresponding to ``r_layers`` (one per layer).
            Entries can be pymiediff material objects or scalar refractive
            indices (converted to ``MatConstant``). This is an alternative to
            ``eps_layers`` for multilayer particles.

        mat_env : pymiediff.materials.Material or float/int/complex/torch.Tensor, optional
            Surrounding (environment) material. Defaults to a refractive index of
            ``1.0`` (air). Scalars are converted to a constant‑index material.

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

        device : str or torch.device, optional
            Torch device on which all tensors will be allocated. If omitted,
            defaults to ``'cpu'``.

        Notes
        -----
        Preferred mode is multilayer (``r_layers`` with ``eps_layers`` or
        ``mat_layers``). Legacy core/shell arguments are kept for backward
        compatibility.

        Examples
        --------
        >>> import pymiediff as pmd
        >>> p = pmd.Particle(r_layers=[50, 80], eps_layers=[2.25, 4.0], mat_env=1.0)
        """
        from pymiediff.materials import MatConstant

        self.device = "cpu" if device is None else device
        self._use_layers = True  # legacy inputs are normalized to layer representation
        self.r_layers = None
        self.eps_layers = None
        self.mat_layers = None

        def _as_material(mat):
            import numpy as np
            if type(mat) in (float, int, complex, torch.Tensor, np.float32, np.float64):
                return MatConstant(mat**2, device=self.device)
            mat.set_device(self.device)
            return mat

        # --- normalize constructor inputs to multilayer representation
        using_layers_api = (r_layers is not None) or (eps_layers is not None) or (mat_layers is not None)
        if using_layers_api:
            if any(v is not None for v in (r_core, mat_core, r_shell, mat_shell)):
                raise ValueError(
                    "Use either multilayer inputs (`r_layers`, `eps_layers`/`mat_layers`) "
                    "or legacy core/shell inputs (`r_core`, `mat_core`, `r_shell`, `mat_shell`)."
                )
            if r_layers is None:
                raise ValueError("`r_layers` must be provided for multilayer mode.")
            if (eps_layers is not None) and (mat_layers is not None):
                raise ValueError("Use either `eps_layers` or `mat_layers`, not both.")
            if (eps_layers is None) and (mat_layers is None):
                raise ValueError("Provide either `eps_layers` or `mat_layers` in multilayer mode.")

            self.r_layers = torch.as_tensor(r_layers, device=self.device)
            if self.r_layers.ndim != 1:
                raise ValueError("For `Particle`, `r_layers` must be one-dimensional (L,).")
            if self.r_layers.numel() < 1:
                raise ValueError("`r_layers` must contain at least one layer.")

            if eps_layers is not None:
                self.eps_layers = torch.as_tensor(eps_layers, device=self.device)
            else:
                if len(mat_layers) != int(self.r_layers.numel()):
                    raise ValueError("`mat_layers` length must match number of `r_layers`.")
                self.mat_layers = [_as_material(mat) for mat in mat_layers]
        else:
            if r_core is None or mat_core is None:
                raise ValueError(
                    "Provide either (`r_layers`, `eps_layers`/`mat_layers`) or legacy "
                    "(`r_core`, `mat_core`)."
                )
            if (r_shell is None) ^ (mat_shell is None):
                raise ValueError(
                    "Either both, or none of shell radius and shell material must be given."
                )

            def _as_radius(val):
                if isinstance(val, torch.Tensor):
                    return val.to(device=self.device)
                return torch.as_tensor(val, device=self.device)

            if r_shell is None:
                self.r_layers = _as_radius(r_core).unsqueeze(0)
                self.mat_layers = [_as_material(mat_core)]
            else:
                self.r_layers = torch.stack((_as_radius(r_core), _as_radius(r_shell)))
                self.mat_layers = [_as_material(mat_core), _as_material(mat_shell)]

            if self.r_layers.ndim != 1:
                raise ValueError("Legacy radii must resolve to scalar values.")
            if self.r_layers.numel() > 1 and not torch.all(self.r_layers[1:] > self.r_layers[:-1]):
                raise ValueError("Layer radii must be strictly increasing.")

        self.mat_env = _as_material(mat_env)
        # if type(mat_env) in (float, int, complex, torch.Tensor):
        #     self.mat_env = MatConstant(mat_env**2, device=self.device)
        # else:
        #     self.mat_env = mat_env
        #     self.mat_env.set_device(self.device)

        # --- compatibility aliases
        self.r_c = self.r_layers[0]
        self.r_s = self.r_layers[-1] if int(self.r_layers.numel()) > 1 else None
        if self.mat_layers is not None:
            self.mat_c = self.mat_layers[0]
            self.mat_s = self.mat_layers[-1] if len(self.mat_layers) > 1 else None
        else:
            self.mat_c = None
            self.mat_s = None

    def set_device(self, device):
        """Move all stored tensors/materials to a new torch device.

        Parameters
        ----------
        device : str or torch.device
            Target device.
        """
        self.device = device

        self.r_layers = self.r_layers.to(device=self.device)
        if self.eps_layers is not None:
            self.eps_layers = self.eps_layers.to(device=self.device)
        if self.mat_layers is not None:
            for mat in self.mat_layers:
                mat.set_device(self.device)
        self.mat_env.set_device(self.device)

        # keep aliases synchronized
        self.r_c = self.r_layers[0]
        self.r_s = self.r_layers[-1] if int(self.r_layers.numel()) > 1 else None
        if self.mat_layers is not None:
            self.mat_c = self.mat_layers[0]
            self.mat_s = self.mat_layers[-1] if len(self.mat_layers) > 1 else None
        else:
            self.mat_c = None
            self.mat_s = None

    def __repr__(self):
        """Return a human-readable summary of particle configuration."""
        out_str = ""
        if int(self.r_layers.numel()) == 1:
            out_str += "homogeneous particle (on device: {})\n".format(self.device)
        else:
            out_str += "multilayer particle (on device: {})\n".format(self.device)
        out_str += " - layers   = {}\n".format(int(self.r_layers.numel()))
        out_str += " - radii    = {}nm\n".format(self.r_layers.data)
        if self.mat_layers is not None:
            out_str += " - materials: {}\n".format([m.__name__ for m in self.mat_layers])
        else:
            out_str += " - epsilon layers provided directly\n"
        out_str += " - environment    : {}\n".format(self.mat_env.__name__)
        return out_str

    def get_material_permittivities(self, k0: torch.Tensor) -> tuple:
        """Evaluate layer and environment permittivities.

        Parameters
        ----------
        k0 : torch.Tensor
            Vacuum wavevector(s), in rad/nm.

        Returns
        -------
        tuple
            ``(eps_layers, eps_env)``.
        """
        k0 = torch.as_tensor(k0, device=self.device)
        wl0 = 2 * torch.pi / k0

        eps_env = self.mat_env.get_epsilon(wavelength=wl0)
        if self.eps_layers is not None:
            eps_layers = self.eps_layers
        else:
            eps_layers = torch.stack(
                [mat.get_epsilon(wavelength=wl0) for mat in self.mat_layers], dim=0
            )

        return eps_layers, eps_env

    def get_mie_coefficients(
        self, k0: torch.Tensor, return_internal=False, **kwargs
    ) -> dict:
        """Compute Mie coefficients for the current particle.

        Parameters
        ----------
        k0 : torch.Tensor
            Vacuum wavevector(s), in rad/nm.
        return_internal : bool, default=False
            Forwarded to backend coefficient solver.
        **kwargs
            Additional keyword arguments passed to
            :func:`pymiediff.multishell.mie_coefficients`.

        Returns
        -------
        dict
            Mie coefficients and metadata.
        """
        from pymiediff.multishell import mie_coefficients

        k0 = torch.as_tensor(k0, device=self.device)
        eps_layers, eps_env = self.get_material_permittivities(k0)
        res = mie_coefficients(
            k0,
            r_layers=self.r_layers,
            eps_layers=eps_layers,
            eps_env=eps_env,
            return_internal=return_internal,
            **kwargs,
        )

        # single particle: remove empty dimension
        from pymiediff.helper.helper import _squeeze_dimensions

        _squeeze_dimensions(res)  # in place

        return res

    def get_cross_sections(self, k0: torch.Tensor, **kwargs) -> dict:
        """Compute spectral cross sections.

        Parameters
        ----------
        k0 : torch.Tensor
            Vacuum wavevector(s), in rad/nm.
        **kwargs
            Forwarded to :func:`pymiediff.multishell.cross_sections`.

        Returns
        -------
        dict
            Cross-section and efficiency spectra.
        """
        from pymiediff.multishell import cross_sections

        k0 = torch.as_tensor(k0, device=self.device)
        eps_layers, eps_env = self.get_material_permittivities(k0)
        res = cross_sections(
            k0,
            r_layers=self.r_layers,
            eps_layers=eps_layers,
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
        """Compute far-field angular scattering quantities.

        Parameters
        ----------
        k0 : torch.Tensor
            Vacuum wavevector(s), in rad/nm.
        theta : torch.Tensor
            Scattering angles in radians.
        **kwargs
            Forwarded to :func:`pymiediff.multishell.angular_scattering`.

        Returns
        -------
        dict
            Angular amplitudes and intensities.
        """
        from pymiediff.multishell import angular_scattering

        k0 = torch.as_tensor(k0, device=self.device)
        theta = torch.as_tensor(theta, device=self.device)

        eps_layers, eps_env = self.get_material_permittivities(k0)
        res_angSca = angular_scattering(
            k0=k0,
            theta=theta,
            r_layers=self.r_layers,
            eps_layers=eps_layers,
            eps_env=eps_env,
            **kwargs,
        )

        # single particle: remove empty dimension
        from pymiediff.helper.helper import _squeeze_dimensions

        res_angSca = _squeeze_dimensions(res_angSca)

        return res_angSca

    def get_nearfields(self, k0: torch.Tensor, r_probe: torch.Tensor, **kwargs) -> dict:
        """Compute near fields at Cartesian probe coordinates.

        Parameters
        ----------
        k0 : torch.Tensor
            Vacuum wavevector(s), in rad/nm.
        r_probe : torch.Tensor
            Probe coordinates of shape ``(N, 3)``.
        **kwargs
            Forwarded to :func:`pymiediff.multishell.nearfields`.

        Returns
        -------
        dict
            Incident, scattered, and total electric/magnetic fields.
        """
        from pymiediff.multishell import nearfields

        k0 = torch.as_tensor(k0, device=self.device)
        r_probe = torch.as_tensor(r_probe, device=self.device)
        assert r_probe.shape[-1] == 3
        assert len(r_probe) == 2

        eps_layers, eps_env = self.get_material_permittivities(k0)
        res_nf = nearfields(
            k0=k0,
            r_probe=r_probe,
            r_layers=self.r_layers,
            eps_layers=eps_layers,
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
