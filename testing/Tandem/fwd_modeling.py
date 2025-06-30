import matplotlib.pyplot as plt
import pymiediff as pmd
import torch
import numpy as np
from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import time

from tqdm import tqdm



# %%

backend = "torch"
device = "cpu"


# %%
# Load data

# Open the file in read mode
with h5py.File("cs_data.h5", "r") as f:
    k0 = torch.from_numpy(f["k0"][:])
    r_c = torch.from_numpy(f["r_c"][:])
    # d_s = torch.from_numpy(f["d_s"][:])
    r_s = torch.from_numpy(f["r_s"][:])
    n_re = torch.from_numpy(f["n_re"][:])
    n_im = torch.from_numpy(f["n_im"][:])
    n = torch.from_numpy(f["n"][:])
    eps_c = torch.from_numpy(f["eps_c"][:])
    eps_s = torch.from_numpy(f["eps_s"][:])

    q_sca = torch.from_numpy(f["q_sca"][:])

wl0 = 2 * torch.pi / k0

print(k0.shape)
print(r_c.shape)
print(eps_c.shape)



# %%
# general config
N_samples = 20000
n_max = 3  # maximum Mie order fixed for performance
eps_env = torch.tensor(1.0, device=device)

lim_r = torch.as_tensor([40, 100], device=device)
lim_n_re = torch.as_tensor([1.5, 3.5], device=device)
lim_n_im = torch.as_tensor([0.0, 0.02], device=device)

lim_q = torch.as_tensor([q_sca.min().item(), q_sca.max().item()], device=device)

print(lim_q)

# %%
plt.plot(q_sca[30].detach().cpu().numpy())

# %%

x = torch.stack((r_c, r_s, n_re[:, 0], n_im[:, 0], n_re[:, 1], n_im[:, 1] ),dim=1)

print(x.shape)
print(q_sca.shape)
dataset = TensorDataset(x, q_sca)

# %%
def nn_pred_to_mie_geometry(pred):
    # implicit normalization: multiply by user-defined limits
    r_c = lim_r.max() * (pred[:, 0])
    r_s = lim_r.max() * (pred[:, 0] + pred[:, 1])
    n_c = lim_n_re.max() * pred[:, 2] + lim_n_im.max() * (1j * pred[:, 3])
    n_s = lim_n_re.max() * pred[:, 4] + lim_n_im.max() * (1j * pred[:, 5])

    eps_c = torch.ones_like(k0).unsqueeze(0) * n_c.unsqueeze(1) ** 2
    eps_s = torch.ones_like(k0).unsqueeze(0) * n_s.unsqueeze(1) ** 2

    return r_c, r_s, eps_c, eps_s


# %%
def nn_pred_to_spectra(pred):
    return lim_q.max() * pred # * (lim_q.max() - lim_q.min()) + lim_q.min()


# %%

class ForwardFullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, num_hidden_layers=2):
        super().__init__()

        # Input layer
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        # Combine into a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# %%

def forward_train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    prog_bar = tqdm(enumerate(dataloader), total=size // dataloader.batch_size)
    for i_batch, (batch_x, batch_y) in prog_bar:
        # model prediction: generate core-shell particles

        pred_y = model(batch_x)

        # calc. loss
        loss = loss_fn(nn_pred_to_spectra(pred_y), batch_y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if i_batch % 100 == 0:
        loss, current = loss.item(), i_batch * dataloader.batch_size + len(batch_x)
        prog_bar.set_description(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# %%
torch.manual_seed(42)

model = ForwardFullyConnected(
    input_dim=6,
    output_dim=len(k0),
    hidden_dim=256,
    num_hidden_layers=3
    ).to(device)

confs = [
    dict(bs=16, lr=1e-3, n_ep=2),
    dict(bs=32, lr=1e-3, n_ep=3),
    dict(bs=64, lr=1e-4, n_ep=5),
    dict(bs=128, lr=1e-4, n_ep=6),
    # dict(bs=256, lr=1e-4, n_ep=10),
]

t_start = time.time()
for conf in confs:
    learning_rate = conf["lr"]
    batch_size = conf["bs"]
    epochs = conf["n_ep"]
    print("-------------------------------")
    print(f"LR={learning_rate}, batch_size={batch_size}")
    print("-------------------------------")

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for t in range(epochs):
        print(f"Epoch {t+1}, time={time.time()-t_start:.2f}s")
        forward_train_loop(train_dataloader, model, loss_fn, optimizer)
print("Done!")



# %%
# test the network
# ----------------
# Do some qualitative tests:
# Let the trained network predict some particle geometries and compare
# their Mie spectra with the traget spectrum.

# pick a few of the training samples for testing.
# Note: Ideally tests should be done on separate samples!
pred = model(x)

# evaluate Mie
pred_q_sca = nn_pred_to_spectra(pred)

# plot
i_plot = np.random.randint(len(x), size=4)
plt.figure(figsize=(12, 10))
for i_n, i in enumerate(i_plot):
    plt.subplot(2, 2, i_n + 1)
    plt.plot(wl0.detach().cpu().numpy(), q_sca[i].detach().cpu().numpy(), label="reference")
    plt.plot(
        wl0.detach().cpu().numpy(),
        pred_q_sca[i].detach().cpu().numpy(),
        label="predicted",
    )
    plt.legend()
    plt.xlabel("wavelength (nm)")
    plt.ylabel("scat. efficiency")
plt.show()
# %%
