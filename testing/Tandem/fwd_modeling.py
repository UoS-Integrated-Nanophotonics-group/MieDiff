import matplotlib.pyplot as plt
import pymiediff as pmd
import torch
import numpy as np
from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py



# %%

backend = "torch"
device = "cpu"


# %%
# Load data

# Open the file in read mode
with h5py.File("data.h5", "r") as f:
    k0 = torch.from_numpy(f["k0"][:])
    r_c = torch.from_numpy(f["r_c"][:])
    # d_s = torch.from_numpy(f["d_s"][:])
    r_s = torch.from_numpy(f["r_s"][:])
    # n_re = torch.from_numpy(f["n_re"][:])
    # n_im = torch.from_numpy(f["n_im"][:])
    # n = torch.from_numpy(f["n"][:])
    eps_c = torch.from_numpy(f["eps_c"][:])
    eps_s = torch.from_numpy(f["eps_s"][:])

    q_sca = torch.from_numpy(f["q_sca"][:])


print(k0.shape)      # torch.Size([40])
print(eps_c.shape)   # torch.Size([20000, 40])
