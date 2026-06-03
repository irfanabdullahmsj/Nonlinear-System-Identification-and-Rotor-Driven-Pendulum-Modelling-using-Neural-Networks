# ============================================================
# RK4 Neural ODE — Rotational System Identification
# ============================================================
# Fixes vs previous version:
#   A. Damping floor: b = softplus(b_raw) + 0.01 — prevents
#      damping collapsing to near-zero and amplitude blow-up
#   B. Long-horizon rollout loss each epoch — trains stability
#      over 3× window length, suppresses phase drift in test
#   C. Frequency-pinned FFT loss — dominant bin weighted 10×
#      to lock the model onto the correct natural frequency
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from scipy.io import loadmat
from scipy.signal import welch

from torch.utils.data import Dataset, DataLoader

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# LOAD DATA
# ============================================================

data = loadmat(
    r"I:\h_da\Summer 2026\Team_Project\Data_Preprocessing\Reduced_Exp_12May_1457_upsamp_filt.mat"
)

phi  = data['PA'].squeeze()
phid = data['PAD'].squeeze()
u    = data['u'].squeeze()
t    = data['t'].squeeze()

# ============================================================
# TIME STEP
# ============================================================

dt = float(t[1] - t[0])
print(f"dt = {dt:.6f} s")

# ============================================================
# NORMALIZATION
# ============================================================

phi_mean  = np.mean(phi)
phi_std   = np.std(phi)
phid_mean = np.mean(phid)
phid_std  = np.std(phid)
u_mean    = np.mean(u)
u_std     = np.std(u)

phi_norm  = (phi  - phi_mean)  / phi_std
phid_norm = (phid - phid_mean) / phid_std
u_norm    = (u    - u_mean)    / u_std

# ============================================================
# DOMINANT FREQUENCY ESTIMATION
# ============================================================

fs = 1.0 / dt
f_psd, psd = welch(
    phi - np.mean(phi),
    fs=fs,
    nperseg=min(4096, len(phi) // 4)
)

f_dom   = f_psd[np.argmax(psd)]
omega_n = 2.0 * np.pi * f_dom

print(f"Dominant frequency = {f_dom:.4f} Hz")
print(f"Natural frequency  = {omega_n:.4f} rad/s")

# ============================================================
# NUMERICALLY STABLE INVERSE SOFTPLUS
# ============================================================

def inv_softplus(x: float) -> float:
    """
    Inverse of softplus(y) = log(1 + exp(y)).
    Safe for large x: when x >= 20, exp(x) >> 1 so result ~= x.
    """
    if x >= 20.0:
        return x
    return float(np.log(np.exp(x) - 1.0))

# ============================================================
# INITIAL PHYSICS PARAMETERS
# ============================================================

# a = -softplus(a_raw)  ->  always negative (restoring stiffness)
a_magnitude = omega_n ** 2 * (phi_std / phid_std)

# Fix A: raise b_init so model starts with meaningful damping
b_init = 0.05   # was 0.01
c_init = 0.10

print("\nInitial parameters:")
print(f"a_magnitude = {a_magnitude:.6f}  (a = -{a_magnitude:.6f})")
print(f"b_init      = {b_init:.6f}")
print(f"c_init      = {c_init:.6f}")

# ============================================================
# LATENT MEMORY STATE
# ============================================================

# z cold-starts at 0 for every window (by design -- within-window latent)
z0 = np.zeros_like(phi_norm)
x  = np.stack((phi_norm, phid_norm, z0), axis=1)

# ============================================================
# TENSORS
# ============================================================

x_tensor = torch.tensor(x,      dtype=torch.float32).to(device)
u_tensor = torch.tensor(u_norm, dtype=torch.float32).unsqueeze(1).to(device)
t_tensor = torch.tensor(t,      dtype=torch.float32).to(device)

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

N = len(t_tensor)

train_idx = int(0.70 * N)
val_idx   = int(0.85 * N)

x_train = x_tensor[:train_idx]
u_train = u_tensor[:train_idx]
t_train = t_tensor[:train_idx]

x_val   = x_tensor[train_idx:val_idx]
u_val   = u_tensor[train_idx:val_idx]
t_val   = t_tensor[train_idx:val_idx]

x_test  = x_tensor[val_idx:]
u_test  = u_tensor[val_idx:]
t_test  = t_tensor[val_idx:]

# ============================================================
# DATASET
# ============================================================

class TrajectoryDataset(Dataset):

    def __init__(self, x_windows, u_windows, t_windows):
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


def build_windows(x_data, u_data, t_data, window_size, stride, dt_val):
    """
    Build overlapping windows with a canonical local time vector
    constructed from dt -- independent of absolute timestamps.
    """
    x_windows = []
    u_windows = []
    t_windows = []

    t_local = torch.linspace(
        0.0,
        (window_size - 1) * dt_val,
        window_size,
        dtype=torch.float32,
        device=x_data.device
    )

    for i in range(0, len(x_data) - window_size, stride):
        x_windows.append(x_data[i : i + window_size])
        u_windows.append(u_data[i : i + window_size])
        t_windows.append(t_local)

    return x_windows, u_windows, t_windows

# ============================================================
# ODE FUNCTION
# ============================================================

class ODEFunc(nn.Module):

    def __init__(self, a_magnitude, b_init, c_init, dt):

        super().__init__()

        self.register_buffer(
            "dt_tensor",
            torch.tensor(dt, dtype=torch.float32)
        )

        # a = -softplus(a_raw), frozen until unfreeze_epoch
        self.a_raw = nn.Parameter(
            torch.tensor(inv_softplus(a_magnitude), dtype=torch.float32),
            requires_grad=False
        )

        # b = softplus(b_raw) + damping_floor  (Fix A)
        self.b_raw = nn.Parameter(
            torch.tensor(inv_softplus(b_init), dtype=torch.float32)
        )

        # Fix A: hard minimum damping -- b can never collapse to ~0
        self.register_buffer(
            "damping_floor",
            torch.tensor(0.01, dtype=torch.float32)
        )

        # Unconstrained input gain
        self.c = nn.Parameter(
            torch.tensor(float(c_init), dtype=torch.float32)
        )

        # Learnable equilibrium offset
        self.phi_eq = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )

        # Residual NN
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )

        self.u_batch = None
        self._residual_accumulator = []

    # ----------------------------------------------------------

    def set_control_batch(self, u_batch):
        self.u_batch = u_batch

    def reset_residual_accumulator(self):
        self._residual_accumulator = []

    def mean_accumulated_residual_sq(self):
        """MSE of residuals accumulated across all RK4 stage evals."""
        if len(self._residual_accumulator) == 0:
            return torch.tensor(0.0, device=self.a_raw.device)
        return torch.stack(self._residual_accumulator).pow(2).mean()

    # ----------------------------------------------------------

    def forward(self, t, x):

        tau  = t / self.dt_tensor
        idx0 = int(min(max(int(tau.item()), 0), self.u_batch.shape[1] - 2))
        idx1 = idx0 + 1

        alpha     = tau - idx0
        u_current = (1.0 - alpha) * self.u_batch[:, idx0, :] \
                  +        alpha  * self.u_batch[:, idx1, :]

        phi   = x[:, 0:1]
        omega = x[:, 1:2]
        z     = x[:, 2:3]

        nn_out   = self.net(torch.cat([phi, omega, z, u_current], dim=1))
        residual = nn_out[:, 0:1]
        dz       = nn_out[:, 1:2]

        self._residual_accumulator.append(residual.detach().mean())

        a = -F.softplus(self.a_raw)
        b =  F.softplus(self.b_raw) + self.damping_floor   # Fix A

        domega = (
            a * (phi - self.phi_eq)
            - b * omega
            + self.c * u_current
            + residual
        )

        return torch.cat([omega, domega, dz], dim=1)

# ============================================================
# MODEL
# ============================================================

func = ODEFunc(a_magnitude, b_init, c_init, dt).to(device)

# ============================================================
# OPTIMIZER FACTORY
# ============================================================

def make_optimizer(func, unfreeze_a=False):
    physics_params = [func.b_raw, func.c, func.phi_eq]

    if unfreeze_a and func.a_raw not in physics_params:
        physics_params.append(func.a_raw)

    return torch.optim.Adam(
        [
            {'params': physics_params,        'lr': 5e-4},
            {'params': func.net.parameters(), 'lr': 1e-3}
        ],
        weight_decay=1e-6,
        eps=1e-7
    )

optimizer = make_optimizer(func, unfreeze_a=False)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ============================================================
# TRAINING SETTINGS
# ============================================================

cycles_target = 4

base_window = max(int(cycles_target / (dt * f_dom)), 100)
max_window  = max(int(8             / (dt * f_dom)), 400)

print(f"\nBase window = {base_window} steps")
print(f"Max  window = {max_window} steps")

epochs              = 100
grow_every          = 8
unfreeze_epoch      = 20
warmup_epochs       = 5
early_stop_patience = 20

best_val         = float('inf')
patience_counter = 0
a_unfrozen       = False

loss_history     = []
val_loss_history = []
param_history    = []

# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(epochs):

    # --------------------------------------------------------
    # Warmup LR
    # --------------------------------------------------------
    if epoch < warmup_epochs:
        scale = (epoch + 1) / warmup_epochs
        optimizer.param_groups[0]['lr'] = 5e-4 * scale
        optimizer.param_groups[1]['lr'] = 1e-3 * scale

    # --------------------------------------------------------
    # Unfreeze stiffness
    # --------------------------------------------------------
    if epoch == unfreeze_epoch and not a_unfrozen:
        func.a_raw.requires_grad_(True)
        optimizer = make_optimizer(func, unfreeze_a=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        a_unfrozen = True
        print(f"\n>>> Epoch {epoch}: Unfreezing stiffness <<<\n")

    func.train()

    current_window = min(
        base_window + (epoch // grow_every) * 50,
        max_window
    )
    stride = current_window // 2

    x_windows, u_windows, t_windows = build_windows(
        x_train, u_train, t_train,
        current_window, stride, dt
    )

    dataset = TrajectoryDataset(x_windows, u_windows, t_windows)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    # Fix C: dominant bin index and weight vector for this window size
    dom_bin     = int(round(f_dom * current_window * dt))
    dom_bin     = max(1, min(dom_bin, current_window // 2))
    fft_weights = torch.ones(current_window // 2 + 1, device=device)
    fft_weights[dom_bin] = 10.0

    total_loss = 0.0

    # --------------------------------------------------------
    # Windowed mini-batch loop
    # --------------------------------------------------------
    for x_batch, u_batch, t_batch in loader:

        optimizer.zero_grad()

        func.reset_residual_accumulator()
        func.set_control_batch(u_batch)

        t_local = t_batch[0]

        pred = odeint(func, x_batch[:, 0, :], t_local, method='rk4')
        pred = pred.permute(1, 0, 2)   # (batch, time, state)

        # State loss
        loss_state = torch.mean((pred - x_batch) ** 2)

        # Derivative consistency loss
        true_dx = x_batch[:, 1:, :] - x_batch[:, :-1, :]
        pred_dx = pred[:, 1:, :]    - pred[:, :-1, :]
        loss_dx = torch.mean((true_dx - pred_dx) ** 2)

        # Fix C: frequency-pinned FFT loss
        fft_true = torch.fft.rfft(x_batch[:, :, 0], dim=1)
        fft_pred = torch.fft.rfft(pred[:, :, 0],    dim=1)
        w        = fft_weights[:fft_true.shape[1]]
        loss_fft = torch.mean(w * torch.abs(fft_true - fft_pred).pow(2))

        # Residual regularisation averaged over all RK4 stages
        loss_residual = func.mean_accumulated_residual_sq()

        loss = (
            loss_state
            + 0.10 * loss_dx
            + 0.05 * loss_fft
            + 1e-4 * loss_residual
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    # --------------------------------------------------------
    # Fix B: long-horizon rollout loss (3x current window)
    # Trains the model to stay stable beyond its training window.
    # --------------------------------------------------------
    func.train()
    optimizer.zero_grad()

    long_len  = min(3 * current_window, train_idx - 1)
    x_long    = x_train[:long_len].unsqueeze(0)
    u_long    = u_train[:long_len].unsqueeze(0)
    t_long    = torch.linspace(
        0.0, (long_len - 1) * dt, long_len,
        dtype=torch.float32, device=device
    )

    func.reset_residual_accumulator()
    func.set_control_batch(u_long)

    pred_long = odeint(func, x_long[:, 0, :], t_long, method='rk4')
    pred_long = pred_long.permute(1, 0, 2)

    loss_long = 0.5 * torch.mean((pred_long - x_long) ** 2)
    loss_long.backward()
    torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
    optimizer.step()

    loss_history.append(avg_loss)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    func.eval()

    with torch.no_grad():

        x_vw, u_vw, t_vw = build_windows(
            x_val, u_val, t_val,
            current_window, stride, dt
        )

        val_losses = []

        for xv, uv, tv in zip(x_vw, u_vw, t_vw):
            xv = xv.unsqueeze(0)
            uv = uv.unsqueeze(0)

            func.reset_residual_accumulator()
            func.set_control_batch(uv)

            pred_v = odeint(func, xv[:, 0, :], tv, method='rk4')
            pred_v = pred_v.permute(1, 0, 2)

            val_losses.append(torch.mean((pred_v - xv) ** 2).item())

        val_loss = float(np.mean(val_losses))

    val_loss_history.append(val_loss)
    scheduler.step(val_loss)

    a_val = -F.softplus(func.a_raw).item()
    b_val =  F.softplus(func.b_raw).item() + func.damping_floor.item()

    param_history.append({
        'a': a_val,
        'b': b_val,
        'c': func.c.item()
    })

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------
    if val_loss < best_val:
        best_val         = val_loss
        patience_counter = 0
        torch.save(
            {
                'model':     func.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch':     epoch,
                'val_loss':  val_loss
            },
            'best_model_rk4.pt'
        )
    else:
        patience_counter += 1

    if patience_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch}")
        break

    if epoch % 5 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Train {avg_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"LongH {loss_long.item():.6f} | "
            f"a={a_val:.5f} | "
            f"b={b_val:.5f} | "
            f"c={func.c.item():.5f}"
        )

# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load('best_model_rk4.pt')
func.load_state_dict(checkpoint['model'])

# Rebuild optimizer to match the checkpoint's frozen/unfrozen state
_ckpt_was_unfrozen = any(
    len(g['params']) > 3
    for g in checkpoint['optimizer']['param_groups']
)
if _ckpt_was_unfrozen and not a_unfrozen:
    func.a_raw.requires_grad_(True)
    a_unfrozen = True

optimizer = make_optimizer(func, unfreeze_a=a_unfrozen)

_saved_groups   = checkpoint['optimizer']['param_groups']
_current_groups = optimizer.param_groups

if len(_saved_groups) == len(_current_groups) and all(
    len(sg['params']) == len(cg['params'])
    for sg, cg in zip(_saved_groups, _current_groups)
):
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    print(
        "Warning: optimizer state not restored "
        "(group size mismatch -- safe to ignore for evaluation)."
    )

print(f"\nBest validation loss = {best_val:.6f}")

# ============================================================
# BURN-IN WARM START
# ============================================================

burnin = min(200, len(x_val) - 1)

x_burnin = x_tensor[val_idx - burnin : val_idx]
u_burnin = u_tensor[val_idx - burnin : val_idx]

t_burnin_local = torch.linspace(
    0.0, (burnin - 1) * dt, burnin,
    dtype=torch.float32, device=device
)

func.eval()

with torch.no_grad():
    func.reset_residual_accumulator()
    func.set_control_batch(u_burnin.unsqueeze(0))

    burnin_pred = odeint(
        func, x_burnin[0].unsqueeze(0),
        t_burnin_local, method='rk4'
    )
    burnin_pred = burnin_pred.permute(1, 0, 2)

x0_test_warmed = burnin_pred[0, -1, :].unsqueeze(0)

# ============================================================
# TEST ROLLOUT
# ============================================================

t_local_test = torch.linspace(
    0.0, (len(t_test) - 1) * dt, len(t_test),
    dtype=torch.float32, device=device
)

func.eval()

with torch.no_grad():
    func.reset_residual_accumulator()
    func.set_control_batch(u_test.unsqueeze(0))

    pred_test = odeint(
        func, x0_test_warmed,
        t_local_test, method='rk4'
    )
    pred_test = pred_test.permute(1, 0, 2)
    pred_test = pred_test.squeeze(0).cpu().numpy()

# ============================================================
# DENORMALIZE
# ============================================================

phi_pred  = pred_test[:, 0] * phi_std  + phi_mean
phid_pred = pred_test[:, 1] * phid_std + phid_mean

phi_true  = phi[val_idx:]
phid_true = phid[val_idx:]
t_true    = t[val_idx:]

# ============================================================
# METRICS
# ============================================================

phi_rmse  = np.sqrt(np.mean((phi_pred  - phi_true)  ** 2))
phid_rmse = np.sqrt(np.mean((phid_pred - phid_true) ** 2))

phi_nrmse  = phi_rmse  / (phi_true.max()  - phi_true.min())  * 100
phid_nrmse = phid_rmse / (phid_true.max() - phid_true.min()) * 100

print("\nTest Metrics:")
print(f"Position RMSE  = {phi_rmse:.4f} deg")
print(f"Velocity RMSE  = {phid_rmse:.4f} deg/s")
print(f"Position NRMSE = {phi_nrmse:.2f}%")
print(f"Velocity NRMSE = {phid_nrmse:.2f}%")

# ============================================================
# PLOTS
# ============================================================

# Loss curves
plt.figure(figsize=(10, 5))
plt.plot(loss_history,     label='Train Loss (windowed)')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Physics parameters
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, key, label in zip(
    axes,
    ['a', 'b', 'c'],
    ['a -- stiffness', 'b -- damping (incl. floor)', 'c -- input gain']
):
    ax.plot([p[key] for p in param_history])
    ax.set_title(label)
    ax.set_xlabel('Epoch')
    ax.grid(True)
plt.suptitle('Physics Parameter Evolution')
plt.tight_layout()

# Angular position
plt.figure(figsize=(12, 5))
plt.plot(t_true, phi_true,        label='True')
plt.plot(t_true, phi_pred,  '--', label='Prediction')
plt.xlabel('Time [s]')
plt.ylabel('Angle [deg]')
plt.title(f'Angular Position  |  NRMSE = {phi_nrmse:.2f}%')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Angular velocity
plt.figure(figsize=(12, 5))
plt.plot(t_true, phid_true,        label='True')
plt.plot(t_true, phid_pred,  '--', label='Prediction')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [deg/s]')
plt.title(f'Angular Velocity  |  NRMSE = {phid_nrmse:.2f}%')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()