"""
Single-precision CPU vs GPU benchmark for the Peña multilayer backend.

This example computes spectra for a 3-layer sphere at 256 wavelengths using
`backend="pena"` and compares runtime on CPU and CUDA (if available).
"""

import time
import torch
import pymiediff as pmd


def _benchmark_cross_sections(device, n_runs=5, n_warmup=2, N_wl=256):
    dtype_r = torch.float32
    dtype_c = torch.complex64
    dtype_r = torch.float64
    dtype_c = torch.complex128

    wl = torch.linspace(450.0, 950.0, N_wl, dtype=dtype_r, device=device)
    k0 = 2.0 * torch.pi / wl

    # 3-layer geometry (nm)
    r_layers = torch.tensor([55.0, 85.0, 120.0], dtype=dtype_r, device=device)

    # Weakly dispersive layer permittivities for benchmarking
    w = wl / 1000.0
    eps_l1 = (2.30 + 0.08 / w**2) ** 2 + 1j * (0.006 + 0.002 * w)
    eps_l2 = (1.85 + 0.05 / w**2) ** 2 + 1j * (0.004 + 0.001 * w)
    eps_l3 = (1.55 + 0.03 / w**2) ** 2 + 1j * (0.002 + 0.001 * w)
    eps_layers = torch.stack((eps_l1, eps_l2, eps_l3), dim=0).to(dtype=dtype_c)
    eps_env = torch.tensor((1.33**2) + 0j, dtype=dtype_c, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # warmup
    for _ in range(n_warmup):
        _ = pmd.multishell.cross_sections(
            k0=k0,
            r_layers=r_layers,
            eps_layers=eps_layers,
            eps_env=eps_env,
            backend="pena",
            precision="single",
            n_max=None,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    res = None
    for _ in range(n_runs):
        res = pmd.multishell.cross_sections(
            k0=k0,
            r_layers=r_layers,
            eps_layers=eps_layers,
            eps_env=eps_env,
            backend="pena",
            precision="single",
            n_max=None,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0

    return dt / n_runs, res


if __name__ == "__main__":
    N_wl = 500000
    print(f"Benchmark CPU vs GP:, single precision, 3 layers, {N_wl} wavelengths")

    cpu = torch.device("cpu")
    t_cpu, res_cpu = _benchmark_cross_sections(cpu, N_wl=N_wl)
    print(f"CPU  : {t_cpu*1e3:8.2f} ms / call")

    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        t_gpu, res_gpu = _benchmark_cross_sections(gpu, N_wl=N_wl)
        speedup = t_cpu / t_gpu
        print(f"GPU  : {t_gpu*1e3:8.2f} ms / call")
        print(f"Speedup (CPU/GPU): {speedup:.2f}x")

        # quick numerical consistency check
        qext_cpu = res_cpu["q_ext"].detach().cpu()
        qext_gpu = res_gpu["q_ext"].detach().cpu()
        rel = (qext_cpu - qext_gpu).abs() / qext_cpu.abs().clamp_min(1e-6)
        print(f"Max relative difference in q_ext: {rel.max().item():.3e}")
    else:
        print("CUDA not available on this system. GPU benchmark skipped.")
