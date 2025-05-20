# %%
# imports
# -------

import matplotlib.pyplot as plt
import pymiediff as pmd
import torch
import numpy as np

# %%
device = torch.device("cpu")  # "cuda:0" if torch.cuda.is_available() else
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
lim_r = torch.as_tensor([40, 100], dtype=torch.float32, device=device)
lim_n_re = torch.as_tensor([2.0, 4.5], dtype=torch.float32, device=device)
lim_n_im = torch.as_tensor([0.0, 0.1], dtype=torch.float32, device=device)

# %%


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
    r_c_n = (r_c - lim_r.min()) / (lim_r.max() - lim_r.min())
    d_s_n = (d_s - lim_r.min()) / (lim_r.max() - lim_r.min())

    r_opt = torch.stack((r_c_n, d_s_n))

    n_c = eps_c**0.5
    n_s = eps_s**0.5

    n_c_re = n_c.real
    n_c_im = n_c.imag
    n_s_re = n_s.real
    n_s_im = n_s.imag

    # core and shell complex ref. index
    n_c_re_n = (n_c_re - lim_n_re.min()) / (lim_n_re.max() - lim_n_re.min())
    n_c_im_n = (n_c_im - lim_n_im.min()) / (lim_n_im.max() - lim_n_im.min())
    n_s_re_n = (n_s_re - lim_n_re.min()) / (lim_n_re.max() - lim_n_re.min())
    n_s_im_n = (n_s_im - lim_n_im.min()) / (lim_n_im.max() - lim_n_im.min())

    n_opt = torch.stack((n_c_re_n, n_s_re_n, n_c_im_n, n_s_im_n))

    return r_opt, n_opt


# %%

x = torch.from_numpy(np.load("x.npy")).type(torch.float32).to(device)
y = torch.from_numpy(np.load("y.npy")).type(torch.float32).to(device)

print(x.dtype)

x_meta = torch.from_numpy(np.loadtxt("x_meta.txt")).type(torch.float32).to(device)
y_meta = torch.from_numpy(np.loadtxt("y_meta.txt")).type(torch.float32).to(device)

lim_q = y_meta[:2]

print(lim_q)


# %%


def spectra_to_normlaised(spectra):
    return (spectra - lim_q.min()) / (lim_q.max() - lim_q.min())


def spectra_to_physical(spectra_n):
    return spectra_n * (lim_q.max() - lim_q.min()) + lim_q.min()


# %%
# train test split
# ----------------
print(x.shape[1])
slpit_indx = int(0.8 * x.shape[1])  # about 80 : 20 split

X_train, X_val = x[:, :slpit_indx].T, x[:, slpit_indx:].T
y_train, y_val = y[:, :slpit_indx].T, y[:, slpit_indx:].T


print(X_train.shape)


# %%
hidden_dim_forward = 512
num_layers_forward = 3
spectra_dim = 32
design_dim = 6


# %%
import torch.nn as nn
from dl_models import FullyConnected, FullyConnectedResNet

# %%


# %%
num_epochs = 200

from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def objective(trial):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 256 * 2])
    num_layers = trial.suggest_categorical("num_layers", [2, 3])
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    patience = trial.suggest_categorical("patience ", [16, 32, 64])
    factor = trial.suggest_categorical("factor ", [1 / 3, 1 / 6, 1 / 10])
    # lr = 0.0016628987455025884
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.MSELoss()
    loss_forward = []

    loaded_forward_nn = FullyConnected(
        input_dim=design_dim,
        output_dim=spectra_dim,
        hidden_dim=hidden_dim_forward,
        num_blocks=num_layers_forward,
    )

    loaded_forward_nn.load_state_dict(
        torch.load(
            "2025-05-06-19-47-54_forward_model.pt",
            weights_only=True,
        )
    )

    for param in loaded_forward_nn.parameters():
        param.requires_grad = False

    inverse_nn = FullyConnected(
        input_dim=spectra_dim,
        output_dim=design_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_layers,
    ).to(device)

    optimizer = optim.Adam(inverse_nn.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=patience, factor=factor
    )

    for epoch in range(num_epochs):
        inverse_nn.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            # Forward pass

            x_pred = inverse_nn(batch_y)

            y_pred = loaded_forward_nn(x_pred)

            loss = criterion(y_pred, batch_y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation
        inverse_nn.eval()
        with torch.no_grad():
            val_X = X_val.to(device)
            val_y = y_val.to(device)

            x_pred_val = inverse_nn(val_y)

            y_pred_val = loaded_forward_nn(x_pred_val)

            val_loss = criterion(y_pred_val, val_y)
            scheduler.step(val_loss)

        loss_forward.append([avg_loss, val_loss.detach().item()])

    return val_loss.item()


#  %%
import optuna

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best trial:")
print("  Value: ", study.best_trial.value)
print("  Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# %%

hidden_dim = study.best_trial.params["hidden_dim"]
num_layers = study.best_trial.params["num_layers"]
patience = study.best_trial.params["patience "]
factor = study.best_trial.params["factor "]
lrate = study.best_trial.params["lr"]


print(hidden_dim, num_layers, lrate)

# %%
num_epochs = 500

from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


criterion = nn.MSELoss()
loss_forward = []


loaded_forward_nn = FullyConnected(
    input_dim=design_dim,
    output_dim=spectra_dim,
    hidden_dim=hidden_dim_forward,
    num_blocks=num_layers_forward,
)

loaded_forward_nn.load_state_dict(
    torch.load(
        "2025-05-06-19-47-54_forward_model.pt",
        weights_only=True,
    )
)

for param in loaded_forward_nn.parameters():
    param.requires_grad = False


inverse_nn = FullyConnected(
    input_dim=spectra_dim,
    output_dim=design_dim,
    hidden_dim=hidden_dim,
    num_blocks=num_layers,
).to(device)

optimizer = optim.Adam(inverse_nn.parameters(), lr=lrate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=patience, factor=factor
)

for epoch in range(num_epochs):
    inverse_nn.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        # Forward pass

        x_pred = inverse_nn(batch_y)

        y_pred = loaded_forward_nn(x_pred)

        loss = criterion(y_pred, batch_y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # Validation
    inverse_nn.eval()
    with torch.no_grad():
        val_X = X_val.to(device)
        val_y = y_val.to(device)

        x_pred_val = inverse_nn(val_y)

        y_pred_val = loaded_forward_nn(x_pred_val)

        val_loss = criterion(y_pred_val, val_y)

        scheduler.step(val_loss)

    loss_forward.append([avg_loss, val_loss.detach().item()])

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.10f}, Val Loss: {val_loss.item():.10f}"
        )


# %%
loss_forward = np.array(loss_forward)
plt.plot(loss_forward[:, 0])
plt.plot(loss_forward[:, 1])
plt.yscale("log")
plt.show()


# %%
import numpy as np
from datetime import datetime

time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
np.savetxt(f"{time}_inverse_model_loss__hl_{512}_{2}.txt", loss_forward, delimiter=",")
torch.save(inverse_nn.state_dict(), f"{time}_inverse__hl_{512}_{2}.pt")
print(f"model and loss saved at: {time}")

# %%


# %%
num_epochs = 500

from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# Suggest hyperparameters

criterion = nn.MSELoss()
loss_forward = []


loaded_forward_nn = FullyConnected(
    input_dim=design_dim,
    output_dim=spectra_dim,
    hidden_dim=hidden_dim_forward,
    num_blocks=num_layers_forward,
)

loaded_forward_nn.load_state_dict(
    torch.load(
        "2025-05-06-19-47-54_forward_model.pt",
        weights_only=True,
    )
)

for param in loaded_forward_nn.parameters():
    param.requires_grad = False


inverse_nn = FullyConnected(
    input_dim=spectra_dim,
    output_dim=design_dim,
    hidden_dim=hidden_dim,
    num_blocks=num_layers,
).to(device)

optimizer = optim.Adam(inverse_nn.parameters(), lr=lrate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=patience, factor=factor
)

for epoch in range(num_epochs):
    inverse_nn.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        # Forward pass

        x_pred = inverse_nn(batch_y)
        y_pred = []
        for o in range(0, x_pred.shape[0]):

            r_n_norm = x_pred[o, :]

            r_norm = r_n_norm[:2]
            n_norm = r_n_norm[2:]

            r_c_phys, eps_c_phys, r_s_phys, eps_s_phys = params_to_physical(
                r_norm, n_norm
            )

            spectra_phys = pmd.farfield.cross_sections(
                k0, r_c_phys, eps_c_phys, r_s_phys, eps_s_phys, n_max=n_max
            )["q_sca"].type(torch.float32)

            y_pred.append(spectra_to_normlaised(spectra_phys))

        y_pred = torch.stack(y_pred)

        loss = criterion(y_pred, batch_y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # Validation
    inverse_nn.eval()
    with torch.no_grad():
        val_X = X_val.to(device)
        val_y = y_val.to(device)

        x_pred_val = inverse_nn(val_y)

        y_pred_val = []
        for oo in range(0, x_pred_val.shape[0]):

            r_n_norm = x_pred_val[oo, :]

            r_norm = r_n_norm[:2]
            n_norm = r_n_norm[2:]

            r_c_phys, eps_c_phys, r_s_phys, eps_s_phys = params_to_physical(
                r_norm, n_norm
            )

            spectra_phys = pmd.farfield.cross_sections(
                k0, r_c_phys, eps_c_phys, r_s_phys, eps_s_phys, n_max=n_max
            )["q_sca"].type(torch.float32)

            y_pred_val.append(spectra_to_normlaised(spectra_phys))

        y_pred_val = torch.stack(y_pred_val)

        val_loss = criterion(y_pred_val, val_y)

        scheduler.step(val_loss)

    loss_forward.append([avg_loss, val_loss.detach().item()])

    
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.10f}, Val Loss: {val_loss.item():.10f}"
    )

# %%
loss_forward = np.array(loss_forward)
plt.plot(loss_forward[:, 0])
plt.plot(loss_forward[:, 1])
plt.yscale("log")
plt.show()

# %%
import numpy as np
from datetime import datetime

time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
np.savetxt(f"{time}_inverse_model_loss__hl_{512}_{2}_pmd.txt", loss_forward, delimiter=",")
torch.save(inverse_nn.state_dict(), f"{time}_inverse__hl_{512}_{2}_pmd.pt")
print(f"model and loss saved at: {time}")

# %%
