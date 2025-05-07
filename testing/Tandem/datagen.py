# %%
# imports
# -------

import matplotlib.pyplot as plt
import pymiediff as pmd
import torch
import numpy as np

# %%
device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else 
print(device)

# %%
# setup data genaration
# ---------------------



# - define the range of wavelengths to be incuded in data generation
wl_res = 32
wl0 = torch.linspace(400, 800, wl_res).to(device)
k0 = 2 * torch.pi / wl0
n_max = 3
# - constants
n_env = torch.tensor(1.0, device=device)

# - set limits to particle's properties, in this example we limit to dielectric materials
lim_r = torch.as_tensor([40, 100], dtype=torch.float, device=device)
lim_n_re = torch.as_tensor([2.0, 4.5], dtype=torch.float, device=device)
lim_n_im = torch.as_tensor([0.0, 0.1], dtype=torch.float, device=device)


sample_num = 10000

# # Define linspaces from 0 to 1
# param1 = np.linspace(0, 1, 8)
# param2 = np.linspace(0, 1, 8)
# param3 = np.linspace(0, 1, 8)
# param4 = np.linspace(0, 1, 8)

# # Get all combinations of the parameters (shape: [10000, 4])
# param_combinations = np.array(np.meshgrid(param1, param2, param3, param4, indexing='ij')).T.reshape(-1, 4)

# # Convert to PyTorch tensor and transpose to match shape [4, N]
# n_arr = torch.tensor(param_combinations, dtype=torch.double, device=device).T  # Shape: (4, 10000)

# param5 = np.linspace(0, 1, 64)
# param6 = np.linspace(0, 1, 64)

# r_combinations = np.array(np.meshgrid(param5, param6, indexing='ij')).T.reshape(-1, 2)
# r_arr = torch.tensor(r_combinations, dtype=torch.double, device=device).T  # Shape: (2, 10000)



r_arr = torch.tensor(np.random.random((2, sample_num)), dtype=torch.float)
n_arr = torch.tensor(np.random.random((4, sample_num)), dtype=torch.float)

print(n_arr.shape)
print(r_arr.shape)

# %%
# define parameter scaling functions
# ----------------------------------

def params_to_physical(r_opt, n_opt):
    """converts normalised parameters to physical

    Args:
        r_opt (torch.Tensor): normalised radii
        n_opt (torch.Tensor): normalised materials

    Returns:
        torch.Tensor: physical parameters
    """
    
    r_c_n, d_s_n = r_opt
    n_c_re_n, n_s_re_n, n_c_im_n, n_s_im_n = n_opt

    # scale parameters to physical units
    # size parameters
    r_c = r_c_n * (lim_r.max() - lim_r.min()) + lim_r.min()
    d_s = d_s_n * (lim_r.max() - lim_r.min()) + lim_r.min()
    r_s = r_c + d_s
    
    # core and shell complex ref. index
    n_c = (n_c_re_n * (lim_n_re.max() - lim_n_re.min()) + lim_n_re.min()) + 1j * (
        n_c_im_n * (lim_n_im.max() - lim_n_im.min()) + lim_n_im.min()
    )
    n_s = (n_s_re_n * (lim_n_re.max() - lim_n_re.min()) + lim_n_re.min()) + 1j * (
        n_s_im_n * (lim_n_im.max() - lim_n_im.min()) + lim_n_im.min()
    )

    return r_c, n_c**2, r_s, n_s**2



def params_to_normlaised(r_c, eps_c, r_s, eps_s):
    """normalises physical parameters

    Args:
        r_c (torch.Tensor): core raduis
        eps_c (torch.Tensor): complex core eps
        r_s (torch.Tensor): shell raduis
        eps_s (torch.Tensor): complex shell eps

    Returns:
        torch.Tensor: normalised parameters
    """
    d_s = r_s - r_c
    r_c_n = (r_c - lim_r.min())/ (lim_r.max() - lim_r.min())
    d_s_n = (d_s - lim_r.min())/ (lim_r.max() - lim_r.min()) 
    
    r_opt = torch.stack((r_c_n, d_s_n))

    n_c = eps_c**0.5
    n_s = eps_s**0.5

    n_c_re = n_c.real
    n_c_im = n_c.imag
    n_s_re = n_s.real
    n_s_im = n_s.imag

    # core and shell complex ref. index
    n_c_re_n = (n_c_re - lim_n_re.min())/ (lim_n_re.max() - lim_n_re.min())
    n_c_im_n = (n_c_im - lim_n_im.min())/ (lim_n_im.max() - lim_n_im.min()) 
    n_s_re_n = (n_s_re - lim_n_re.min())/ (lim_n_re.max() - lim_n_re.min())
    n_s_im_n = (n_s_im - lim_n_im.min())/ (lim_n_im.max() - lim_n_im.min()) 

    n_opt = torch.stack((n_c_re_n, n_s_re_n, n_c_im_n, n_s_im_n))

    return r_opt, n_opt

# %%
# generate data
# -------------

r_c, eps_c, r_s, eps_s = params_to_physical(r_arr, n_arr)

q_sca = []    
q_abs = []
q_ext = []


for i in range(sample_num):
    args = (k0.detach().cpu(), r_c[i].detach().cpu(), eps_c[i].detach().cpu(), r_s[i].detach().cpu(), eps_s[i].detach().cpu().cpu())
    result = pmd.farfield.cross_sections(*args, n_max=n_max)
    q_sca.append(result["q_sca"])
    q_abs.append(result["q_abs"])
    q_ext.append(result["q_ext"])

q_sca = torch.stack(q_sca).to(device)
q_abs = torch.stack(q_abs).to(device)
q_ext = torch.stack(q_ext).to(device)


# %%
# define spectra scaling functions
# --------------------------------

lim_q = torch.as_tensor([q_sca.min().item(), q_sca.max().item()], dtype=torch.float, device=device)

print(lim_q)

def spectra_to_normlaised(spectra):
    return (spectra - lim_q.min())/ (lim_q.max() - lim_q.min())

def spectra_to_physical(spectra_n):
    return spectra_n * (lim_q.max() - lim_q.min()) + lim_q.min()


# %%
# make x and y datasets
# ---------------------

x = torch.cat((r_arr, n_arr), dim=0)
y = spectra_to_normlaised(q_sca).T

print("X shape:", x.shape)
print("y shape:", y.shape)

x_meta = torch.cat([lim_r, lim_n_re, lim_n_im])
y_meta = torch.cat([lim_q, wl0])

print("x metadata", x_meta)
print("y metadata", y_meta)

# %%
# save datasets to npy

np.save("x.npy", x)
np.save("y.npy", y)

np.savetxt("x_meta.txt", x_meta)
np.savetxt("y_meta.txt", y_meta)


# %%