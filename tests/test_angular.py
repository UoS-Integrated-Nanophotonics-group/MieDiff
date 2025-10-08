# encoding=utf-8
import unittest
import warnings

import numpy as np
import torch

import pymiediff as pmd


class TestForward(unittest.TestCase):
    def test_vsh(self):
        try:
            from miepython.vsh import (
                M_odd_array,
                M_even_array,
                N_odd_array,
                N_even_array,
            )
        except ImportError:
            warnings.warn("`miepython` seems not installed, skipping VSH test.")
            return

        def compute_vsh_miepython(nmax, wavelength, diameter, m_index, r, theta, phi):
            """
            Compute vector spherical harmonics from miepython for orders 1..nmax.

            requires miepython v3.0.2
            
            Args:
            nmax: maximum multipole order (int)
            wavelength: vacuum wavelength λ₀
            diameter: diameter of the sphere d_sphere
            m_index: (complex) refractive index of the sphere relative to medium
            r, theta, phi: arrays (or scalars) of same shape for sampling points

            Returns:
            dict with keys:
                M_odd, M_even, N_odd, N_even — each is an array
                of shape (nmax, *point_shape, 3) corresponding to (r, θ, φ) components
            """
            # We will loop over multipole orders and build arrays
            # Make sure r, theta, phi are numpy arrays
            r = np.asarray(r)
            theta = np.asarray(theta)
            phi = np.asarray(phi)
            pts_shape = r.shape

            # For odd / even M and N, miepython provides “array” versions:
            # M_odd_array(n, λ₀, d_sphere, m_index, r, θ, φ) returns (3, n) arrays (r,θ,φ) × orders
            # But its output format is slightly different; we reshape to (n, *pts_shape, 3)
            # Actually M_odd_array returns 3 arrays (for r, θ, φ) of shape (n,), for a single point.
            # If you have multiple points, you must call per point or vectorize.

            # For vectorization, do a loop over points:
            npts = r.size
            # Flatten the point arrays
            r_flat = r.reshape(-1)
            theta_flat = theta.reshape(-1)
            phi_flat = phi.reshape(-1)

            # Storage arrays
            M_odd = np.zeros((nmax, npts, 3), dtype=complex)
            M_even = np.zeros_like(M_odd)
            N_odd = np.zeros_like(M_odd)
            N_even = np.zeros_like(M_odd)

            # For each point, compute all orders
            for i in range(npts):
                rr = r_flat[i]
                tt = theta_flat[i]
                pp = phi_flat[i]
                # The miepython vsh functions take single r, θ, φ at time
                mo = M_odd_array(nmax, wavelength, diameter, m_index, rr, tt, pp)
                me = M_even_array(nmax, wavelength, diameter, m_index, rr, tt, pp)
                no = N_odd_array(nmax, wavelength, diameter, m_index, rr, tt, pp)
                ne = N_even_array(nmax, wavelength, diameter, m_index, rr, tt, pp)
                # mo etc return arrays of shape (3, nmax) in the order [r, θ, φ] × order
                # Reformat to (nmax, 3)
                # Note: in miepython code, the array is returned as [comp][order]
                for n in range(nmax):
                    M_odd[n, i, :] = np.array([mo[0][n], mo[1][n], mo[2][n]])
                    M_even[n, i, :] = np.array([me[0][n], me[1][n], me[2][n]])
                    N_odd[n, i, :] = np.array([no[0][n], no[1][n], no[2][n]])
                    N_even[n, i, :] = np.array([ne[0][n], ne[1][n], ne[2][n]])

            # Finally reshape point dimension back to original shape
            def reshape_out(arr):
                # arr: (nmax, npts, 3) → (nmax, *pts_shape, 3)
                return arr.reshape((nmax,) + pts_shape + (3,))

            return {
                "M_odd": reshape_out(M_odd),
                "M_even": reshape_out(M_even),
                "N_odd": reshape_out(N_odd),
                "N_even": reshape_out(N_even),
            }

        # - setup some VSH testcases
        n_max_np = 5
        wl0_np = 650.0

        # numpy args
        diameter_np = 0.01  # miepython: test scattered field
        r_np = np.array([200.0, 250.0, 300.5])
        theta_np = np.array([0.5, 2.2, 4.5])
        phi_np = np.array([1.5, 0.2, 2.5])

        # torch args
        k0_torch = 2 * torch.pi / torch.as_tensor(wl0_np)
        r_torch = torch.as_tensor(r_np).unsqueeze(0).unsqueeze(-1)
        theta_torch = torch.as_tensor(theta_np).unsqueeze(0).unsqueeze(-1)
        phi_torch = torch.as_tensor(phi_np).unsqueeze(0).unsqueeze(-1)

        # - evaluate
        for _n_env_test in [1.0, 1.5, 2.0]:
            m_index_np = _n_env_test
            m_index_torch = torch.as_tensor(m_index_np).unsqueeze(0)

            vsh_np = compute_vsh_miepython(
                n_max_np, wl0_np, diameter_np, m_index_np, r_np, theta_np, phi_np
            )
            _vsh_torch = pmd.special.vsh(
                n_max_np,
                k0_torch,
                m_index_torch,
                r_torch,
                theta_torch,
                phi_torch,
                kind=3,
            )
            vsh_torch = {
                "M_odd": _vsh_torch[0],
                "M_even": _vsh_torch[1],
                "N_odd": _vsh_torch[2],
                "N_even": _vsh_torch[3],
            }

            for which_vsh in vsh_torch.keys():
                np.testing.assert_allclose(
                    vsh_torch[which_vsh].numpy(), vsh_np[which_vsh], rtol=1e-5
                )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
