# encoding: utf-8
"""
tandem model
=========================



author: O. Jackson, 03/2025
"""
# %%
# imports
# -------

import matplotlib.pyplot as plt
import pymiediff as pmd
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# %%
# setup data genaration
# ---------------------

# - define the range of wavelengths to be incuded in data generation
wl_res = 25
wl0 = torch.linspace(400, 800, wl_res)
k0 = 2 * torch.pi / wl0
n_max = 3
# - constants
n_env = 1.0

# - set limits to particle's properties, in this example we limit to dielectric materials
lim_r = torch.as_tensor([10, 100], dtype=torch.double)
lim_n_re = torch.as_tensor([1, 4.5], dtype=torch.double)
lim_n_im = torch.as_tensor([0, 0.1], dtype=torch.double)


sample_num = 1000

r_arr = torch.tensor(np.random.random((2, sample_num)), dtype=torch.double)
n_arr = torch.tensor(np.random.random((4, sample_num)), dtype=torch.double)

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

# Check scalers
# r_norm = r_arr[:,0]
# n_norm = n_arr[:,0]

# print("normalised r and n:")
# print(r_norm, n_norm)

# r_c_phys, eps_c_phys, r_s_phys, eps_s_phys = params_to_physical(r_norm, n_norm)

# print("physical r and n:")
# print(r_c_phys, eps_c_phys, r_s_phys, eps_s_phys)

# r_norm , n_norm = params_to_normlaised(r_c_phys, eps_c_phys, r_s_phys, eps_s_phys)

# print("renormalised r and n:")
# print(r_norm, n_norm)


# %%
# generate data
# -------------

r_c, eps_c, r_s, eps_s = params_to_physical(r_arr, n_arr)

q_sca = []    
q_abs = []
q_ext = []


for i in range(sample_num):
    args = (k0, r_c[i], eps_c[i], r_s[i], eps_s[i])
    result = pmd.farfield.cross_sections(*args, n_max=n_max)
    q_sca.append(result["q_sca"])
    q_abs.append(result["q_abs"])
    q_ext.append(result["q_ext"])

q_sca = torch.stack(q_sca)
q_abs = torch.stack(q_abs)
q_ext = torch.stack(q_ext)



# %%
# define spectra scaling functions
# --------------------------------

lim_q = torch.as_tensor([q_sca.min().item(), q_sca.max().item()], dtype=torch.double)

print(lim_q)

def spectra_to_normlaised(spectra):
    return (spectra - lim_q.min())/ (lim_q.max() - lim_q.min())

def spectra_to_physical(spectra_n):
    return spectra_n * (lim_q.max() - lim_q.min()) + lim_q.min()

# Check scalers
# q_sca_phys = q_sca[:,0]

# print("physical spectra:")
# print(q_sca_phys)

# q_sca_norm = spectra_to_normlaised(q_sca_phys)

# print("normalised spectra:")
# print(q_sca_norm)

# q_sca_phys = spectra_to_physical(q_sca_norm)

# print("re-physical spectra:")
# print(q_sca_phys)


# %%
#

x = torch.cat((r_arr, n_arr), dim=0)
y = spectra_to_normlaised(q_sca).T


print("X shape:", x.shape)
print("y shape:", y.shape)

# %%
# train test split
# ----------------

slpit_indx = int(0.8*sample_num) # about 80 : 20 split

X_train, X_val = x[:,:slpit_indx].T, x[:,slpit_indx:].T
y_train, y_val = y[:,:slpit_indx].T, y[:,slpit_indx:].T

# Checking shapes:
# print(slpit_indx)
# print("X train shape:", X_train.shape)
# print("y train shape:", y_train.shape)
# print("X val shape:", X_val.shape)
# print("y val shape:", y_val.shape)

# print(X_train[:,2:].shape)

# print((X_train[0, :])[:2])

# %%
# machine learning config
# -----------------------

from torch import nn
import torch.optim as optim

input_dim=wl_res 
output_dim=6
hidden_dim=128

inverse_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(inverse_nn.parameters(), lr=0.01)

sigmoid = nn.Sigmoid()

# %%
# define and run training loop
# ----------------------------

num_epochs = 100
batch_size = 32

inverse_nn.double()

for epoch in range(num_epochs):
    inverse_nn.train()
    
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        # Get batch
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # Forward pass

        x_pred = inverse_nn(batch_y)

        y_pred = []
        # print(x_pred.shape)
        # print(x_pred.shape[0])
        # print(x_pred.shape[1])
        for o in range(0, x_pred.shape[0]):
            
            r_n_norm = sigmoid(x_pred[o, :])

            r_norm = r_n_norm[:2]
            n_norm = r_n_norm[2:]

            r_c_phys, eps_c_phys, r_s_phys, eps_s_phys = params_to_physical(r_norm, n_norm)

            spectra_phys = pmd.farfield.cross_sections(k0, r_c_phys, eps_c_phys, r_s_phys, eps_s_phys, n_max=n_max)["q_sca"]

            y_pred.append(spectra_phys)#spectra_to_normlaised(spectra_phys))
        
        y_pred = torch.stack(y_pred)


        loss = criterion(y_pred, spectra_to_physical(batch_y)) + criterion(x_pred, batch_X)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    inverse_nn.eval()
    with torch.no_grad():

        x_pred_val = inverse_nn(y_val)

        y_pred_val = []
        for oo in range(0, x_pred_val.shape[0]):
            
            r_n_norm = sigmoid(x_pred_val[oo, :])

            r_norm = r_n_norm[:2]
            n_norm = r_n_norm[2:]

            r_c_phys, eps_c_phys, r_s_phys, eps_s_phys = params_to_physical(r_norm, n_norm)

            spectra_phys = pmd.farfield.cross_sections(k0, r_c_phys, eps_c_phys, r_s_phys, eps_s_phys, n_max=n_max)["q_sca"]

            y_pred_val.append(spectra_phys)#spectra_to_normlaised(spectra_phys))
        
        y_pred_val = torch.stack(y_pred_val)


        val_loss = criterion(y_pred_val, spectra_to_physical(y_val))
    
    # if (epoch+1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
# %%
