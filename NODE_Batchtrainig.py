# ============================================================
# Neural ODE for Rotational System Identification
# Train / Validation / Test Split Version
# ============================================================

from math import tau

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchdiffeq import odeint
from scipy.io import loadmat

from torch.utils.data import Dataset, DataLoader

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# ============================================================
# LOAD DATA
# ============================================================

data = loadmat(
    r"I:\h_da\Summer 2026\Team_Project\Data_Preprocessing\Reduced_Exp_12May_1457_upsamp_filt.mat"
)

phi = data['PA'].squeeze()
phid = data['PAD'].squeeze()
u = data['u'].squeeze()
t = data['t'].squeeze()

# ============================================================
# VISUALIZE RAW DATA
# ============================================================

plt.figure()
plt.plot(t, phi)
plt.title("Angular Position")
plt.xlabel("Time [s]")
plt.ylabel("phi")
plt.grid()

plt.figure()
plt.plot(t, phid)
plt.title("Angular Velocity")
plt.xlabel("Time [s]")
plt.ylabel("phid")
plt.grid()

plt.figure()
plt.plot(t, u)
plt.title("Control Input")
plt.xlabel("Time [s]")
plt.ylabel("u")
plt.grid()

# ============================================================
# NORMALIZATION
# ============================================================

phi_mean = np.mean(phi)
phi_std = np.std(phi)

phid_mean = np.mean(phid)
phid_std = np.std(phid)

u_mean = np.mean(u)
u_std = np.std(u)

# ============================================================
# OPTIONAL DOWNSAMPLING
# ============================================================

skip = 1

phi_ = phi[::skip]
phid_ = phid[::skip]
u_ = u[::skip]
t_ = t[::skip]

# ============================================================
# NORMALIZE
# ============================================================

phi_norm = (phi_ - phi_mean) / phi_std
phid_norm = (phid_ - phid_mean) / phid_std
u_norm = (u_ - u_mean) / u_std

# ============================================================
# STATE VECTOR
# x = [phi, phid]
# ============================================================

x = np.stack(
    (phi_norm, phid_norm),
    axis=1
)

# ============================================================
# CONVERT TO TENSORS
# ============================================================

x_tensor = torch.tensor(
    x,
    dtype=torch.float32
).to(device)

u_tensor = torch.tensor(
    u_norm,
    dtype=torch.float32
).unsqueeze(1).to(device)

t_tensor = torch.tensor(
    t_,
    dtype=torch.float32
).to(device)

# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# 70% train
# 15% validation
# 15% test
# ============================================================

N = len(t_tensor)

train_idx = int(0.70 * N)
val_idx = int(0.85 * N)

# ---------------- TRAIN ----------------

x_train = x_tensor[:train_idx]
u_train = u_tensor[:train_idx]
t_train = t_tensor[:train_idx]

# ---------------- VALIDATION ----------------

x_val = x_tensor[train_idx:val_idx]
u_val = u_tensor[train_idx:val_idx]
t_val = t_tensor[train_idx:val_idx]

# ---------------- TEST ----------------

x_test = x_tensor[val_idx:]
u_test = u_tensor[val_idx:]
t_test = t_tensor[val_idx:]

# ============================================================
# TIME STEP
# ============================================================

dt = float(t_[1] - t_[0])

print("dt =", dt)

# ============================================================
# CREATE WINDOWED TRAJECTORIES
# ============================================================

window_size = 500
stride = 50

x_windows = []
u_windows = []
t_windows = []

for i in range(
    0,
    len(x_train) - window_size,
    stride
):

    x_w = x_train[i:i+window_size]

    u_w = u_train[i:i+window_size]

    t_w = t_train[i:i+window_size]

    # local time
    t_w = t_w - t_w[0]

    x_windows.append(x_w)
    u_windows.append(u_w)
    t_windows.append(t_w)

print("Number of windows:", len(x_windows))

# ============================================================
# DATASET
# ============================================================

class TrajectoryDataset(Dataset):

    def __init__(
        self,
        x_windows,
        u_windows,
        t_windows
    ):

        self.x_windows = x_windows
        self.u_windows = u_windows
        self.t_windows = t_windows

    def __len__(self):

        return len(self.x_windows)

    def __getitem__(self, idx):

        return (
            self.x_windows[idx],
            self.u_windows[idx],
            self.t_windows[idx]
        )

# ============================================================
# DATALOADER
# ============================================================

dataset = TrajectoryDataset(
    x_windows,
    u_windows,
    t_windows
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# ============================================================
# NEURAL ODE MODEL
# ============================================================

class ODEFunc(nn.Module):

    def __init__(self):

        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))  # Linear stiffness
        self.b = nn.Parameter(torch.tensor(0.2))  # Linear damping
        self.c = nn.Parameter(torch.tensor(1.0))  # Control input effect
        self.net = nn.Sequential(

            nn.Linear(3, 64),
            nn.SiLU(),

            nn.Linear(64, 64),
            nn.SiLU(),

            nn.Linear(64, 1)
        )

    def set_control_batch(
        self,
        u_batch
    ):

        self.u_batch = u_batch

    def forward(
        self,
        t,
        x
    ):

        # ----------------------------------------------------
        # INTERPOLATED CONTROL INPUT
        # ----------------------------------------------------

        tau = t / dt

        idx0 = torch.floor(tau).long()

        idx1 = torch.clamp(
            idx0 + 1,
            max=self.u_batch.shape[1] - 1
        )

        alpha = (
            tau - idx0.float()
        ).unsqueeze(0).unsqueeze(1)

        u0 = self.u_batch[:, idx0, :]

        u1 = self.u_batch[:, idx1, :]

        u_current = (
            (1 - alpha) * u0
            + alpha * u1
        )

        # ----------------------------------------------------
        # STATES
        # ----------------------------------------------------

        omega = x[:, 1:2]

        # ----------------------------------------------------
        # NETWORK INPUT
        # ----------------------------------------------------

        inp = torch.cat(
            [x, u_current],
            dim=1
        )

        # ----------------------------------------------------
        # LEARNED DYNAMICS
        # ----------------------------------------------------

        domega = (
            -self.a*x[:,0:1]  # Linear stiffness
            -self.b*x[:,1:2]  # Linear damping
            +self.c*u_current  # Control input effect
            + self.net(inp)  # Neural network learns residual dynamics
        )

        dxdt = torch.cat(
            [omega, domega],
            dim=1
        )

        return dxdt
# ============================================================
# CREATE MODEL
# ============================================================

func = ODEFunc().to(device)

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    func.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

# ============================================================
# TRAINING
# ============================================================

epochs = 100

loss_history = []
val_loss_history = []

for epoch in range(epochs):

    func.train()

    total_loss = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for x_batch, u_batch, t_batch in loader:

        optimizer.zero_grad()

        x0_batch = x_batch[:, 0, :]

        func.set_control_batch(u_batch)

        t_local = t_batch[0]

        pred = odeint(
            func,
            x0_batch,
            t_local,
            method='rk4'
        )

        pred = pred.permute(1, 0, 2)

        loss = torch.mean(
            (pred - x_batch) ** 2
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    loss_history.append(avg_loss)

    # ========================================================
    # VALIDATION
    # ========================================================

    func.eval()

    with torch.no_grad():

        u_val_batch = u_val.unsqueeze(0)

        func.set_control_batch(
            u_val_batch
        )

        x0_val = x_val[0].unsqueeze(0)

        pred_val = odeint(
            func,
            x0_val,
            t_val - t_val[0],
            method='rk4'
        )

        pred_val = pred_val.permute(1,0,2)

        val_loss = torch.mean(
            (
                pred_val.squeeze(0) - x_val
            ) ** 2
        )

    val_loss_history.append(
        val_loss.item()
    )

    # ========================================================
    # PRINT
    # ========================================================

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch:4d} | "
            f"Train Loss = {avg_loss:.6f} | "
            f"Val Loss = {val_loss:.6f}"
        )

# ============================================================
# PLOT TRAINING + VALIDATION LOSS
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    loss_history,
    label='Train Loss'
)

plt.plot(
    val_loss_history,
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()
plt.grid()

# ============================================================
# TESTING
# ============================================================

func.eval()

with torch.no_grad():

    u_test_batch = u_test.unsqueeze(0)

    func.set_control_batch(
        u_test_batch
    )

    x0_test = x_test[0].unsqueeze(0)

    pred_test = odeint(
        func,
        x0_test,
        t_test - t_test[0],
        method='rk4'
    )

# ============================================================
# RESHAPE OUTPUT
# ============================================================

pred_test = pred_test.permute(1,0,2)

pred_test = pred_test.squeeze(0).cpu().numpy()

# ============================================================
# DENORMALIZE
# ============================================================

phi_pred = pred_test[:,0]
phid_pred = pred_test[:,1]

phi_pred_denorm = (
    phi_pred * phi_std
    + phi_mean
)

phid_pred_denorm = (
    phid_pred * phid_std
    + phid_mean
)

# ============================================================
# TRUE TEST SIGNALS
# ============================================================

phi_true = phi_[val_idx:]
phid_true = phid_[val_idx:]

t_true = t_[val_idx:]

# ============================================================
# POSITION PLOT
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    t_true,
    phi_true,
    label='True'
)

plt.plot(
    t_true,
    phi_pred_denorm,
    '--',
    label='NODE Prediction'
)

plt.xlabel("Time [s]")
plt.ylabel("Angle")

plt.title("Angular Position Prediction")

plt.legend()

plt.grid()

# ============================================================
# VELOCITY PLOT
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    t_true,
    phid_true,
    label='True Velocity'
)

plt.plot(
    t_true,
    phid_pred_denorm,
    '--',
    label='NODE Velocity'
)

plt.xlabel("Time [s]")
plt.ylabel("Angular Velocity")

plt.title("Angular Velocity Prediction")

plt.legend()

plt.grid()

plt.show()