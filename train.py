"""
train.py
========
Normalise data, build DataLoaders, run the training loop
with curriculum multi-step rollout loss, early stopping,
then save the best model and scalers.

Key changes vs original
-----------------------
1. Curriculum rollout: horizon ramps from 1 → rollout_horizon_max
   over training, forcing the NN to stay stable under its own feedback.
2. Combined loss = one-step MSE  +  lambda_rollout * multi-step MSE
                                 +  lambda_residual_reg * ||residual||²
3. Raw signal arrays (PA, PAD, u) are kept alongside X/Y so the
   rollout loop can build features from simulated states.
"""

import copy
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

import config as cfg
from config import (
    train_file, val_file, test_file, Ts,
    input_size, delays, max_delay, device,
    batch_size, learning_rate, weight_decay,
    epochs, early_stop_patience,
    scheduler_patience, scheduler_factor,
    noise_std_train, grad_clip_norm,
    rollout_horizon_start, rollout_horizon_max, rollout_ramp_epochs,
    lambda_rollout, lambda_residual_reg,
    model_save_path, x_scaler_save_path, y_scaler_save_path,
)
from dataset import load_all_splits, load_raw_signals
from physics import rk4_step
from model  import ImprovedHybridNN


# ============================================================
# LOAD DATA
# ============================================================

splits = load_all_splits(train_file, val_file, test_file, Ts)

X_train, Y_train = splits["train"][0], splits["train"][1]
X_val,   Y_val   = splits["val"][0],   splits["val"][1]
X_test,  Y_test  = splits["test"][0],  splits["test"][1]

# Raw signal arrays needed for rollout loss
PA_train, PAD_train, u_train = load_raw_signals(train_file)
PA_val,   PAD_val,   u_val   = load_raw_signals(val_file)


# ============================================================
# NORMALISATION
# ============================================================

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_sc = x_scaler.fit_transform(X_train)
Y_train_sc = y_scaler.fit_transform(Y_train)

X_val_sc = x_scaler.transform(X_val)
Y_val_sc = y_scaler.transform(Y_val)

X_test_sc = x_scaler.transform(X_test)
Y_test_sc = y_scaler.transform(Y_test)

# Scaler parameters as tensors for in-graph use
x_mean = torch.tensor(x_scaler.mean_,  dtype=torch.float32).to(device)
x_std  = torch.tensor(x_scaler.scale_, dtype=torch.float32).to(device)
y_mean = torch.tensor(y_scaler.mean_,  dtype=torch.float32).to(device)
y_std  = torch.tensor(y_scaler.scale_, dtype=torch.float32).to(device)


# ============================================================
# TENSORS & LOADERS
# ============================================================

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32).to(device)

X_train_t = to_tensor(X_train_sc)
Y_train_t = to_tensor(Y_train_sc)
X_val_t   = to_tensor(X_val_sc)
Y_val_t   = to_tensor(Y_val_sc)
X_test_t  = to_tensor(X_test_sc)
Y_test_t  = to_tensor(Y_test_sc)

# Keep raw signal tensors for rollout
PA_train_t  = to_tensor(PA_train)
PAD_train_t = to_tensor(PAD_train)
u_train_t   = to_tensor(u_train)

PA_val_t    = to_tensor(PA_val)
PAD_val_t   = to_tensor(PAD_val)
u_val_t     = to_tensor(u_val)

# DataLoader: index-based so we can look up raw signals
train_indices = torch.arange(len(X_train_t))
train_loader  = DataLoader(
    TensorDataset(X_train_t, Y_train_t, train_indices),
    batch_size=batch_size,
    shuffle=True,
)


# ============================================================
# BATCHED RK4  (numpy, operates on (B,) arrays)
# ============================================================

def rk4_step_batch(phi: np.ndarray, phid: np.ndarray,
                   u: np.ndarray, Ts: float):
    """
    Vectorised RK4 over a batch of B states.
    All inputs are (B,) numpy arrays; returns (B,), (B,).
    """
    from config import a0, a1, a2

    def f(ph, phd):
        phdd = (-a1 * phd - a0 * np.sin(ph) + (
            -0.00000992 * u**3
            + 0.00105529 * u**2
            - 0.00489381 * u
        )) / a2
        return phd, phdd

    k1p, k1d = f(phi,                   phid)
    k2p, k2d = f(phi + 0.5*Ts*k1p,     phid + 0.5*Ts*k1d)
    k3p, k3d = f(phi + 0.5*Ts*k2p,     phid + 0.5*Ts*k2d)
    k4p, k4d = f(phi +    Ts*k3p,       phid +    Ts*k3d)

    phi_next  = phi  + (Ts/6.) * (k1p + 2*k2p + 2*k3p + k4p)
    phid_next = phid + (Ts/6.) * (k1d + 2*k2d + 2*k3d + k4d)
    return phi_next, phid_next


# ============================================================
# ROLLOUT LOSS  — fully batched, one model() call per step
# ============================================================

def rollout_loss_fn(model, dataset_indices, PA_sig, PAD_sig, u_sig,
                    horizon):
    """
    Runs the full batch of samples through `horizon` steps in
    parallel.  Only `horizon` forward passes through the model
    total (instead of batch_size × horizon).

    dataset_indices : (B,) LongTensor — 0-based rows into X_train.
                      Signal time k = row + max_delay.
    Returns scalar loss (differentiable).
    """
    PA_np  = PA_sig.cpu().numpy()
    PAD_np = PAD_sig.cpu().numpy()
    u_np   = u_sig.cpu().numpy()
    N      = len(PA_np)

    # Map dataset rows → signal times; drop samples too close to end
    k0s = dataset_indices.cpu().numpy().astype(int) + max_delay
    valid = k0s + horizon < N
    k0s = k0s[valid]
    B   = len(k0s)

    if B == 0:
        return torch.zeros(1, device=device, requires_grad=True)

    # ----------------------------------------------------------------
    # Pre-fetch ground-truth future states  shape (B, horizon)
    # ----------------------------------------------------------------
    gt_phi  = np.stack([PA_np [k0s + h + 1] for h in range(horizon)], axis=1)  # (B, H)
    gt_phid = np.stack([PAD_np[k0s + h + 1] for h in range(horizon)], axis=1)

    gt_phi_t  = torch.tensor(gt_phi,  dtype=torch.float32, device=device)
    gt_phid_t = torch.tensor(gt_phid, dtype=torch.float32, device=device)

    # ----------------------------------------------------------------
    # Pre-fetch u signal for each step  shape (B, horizon)
    # ----------------------------------------------------------------
    u_steps = np.stack([u_np[k0s + h] for h in range(horizon)], axis=1)  # (B, H)

    # ----------------------------------------------------------------
    # Rolling state buffers: plain numpy (B,) — updated each step,
    # no autograd needed here; gradients flow only through res.
    # History ring for building delay features: shape (B, max_delay+1)
    # ----------------------------------------------------------------
    # Initialise history with the true signal up to k0
    phi_hist  = np.stack([PA_np [k0s - dly] for dly in delays], axis=1)  # (B, D)
    phid_hist = np.stack([PAD_np[k0s - dly] for dly in delays], axis=1)
    u_hist    = np.stack([u_np  [k0s - dly] for dly in delays], axis=1)

    # Current state (will be overwritten each step)
    phi_cur  = PA_np [k0s].copy()   # (B,)
    phid_cur = PAD_np[k0s].copy()

    total_loss = torch.zeros(1, device=device)

    for h in range(horizon):

        # ---- Build (B, input_size) feature matrix ----
        # Interleave (phi_dly, phid_dly, u_dly) for each delay tap
        feat_np = np.empty((B, input_size), dtype=np.float32)
        for di, dly in enumerate(delays):
            feat_np[:, di*3 + 0] = phi_hist [:, di]
            feat_np[:, di*3 + 1] = phid_hist[:, di]
            feat_np[:, di*3 + 2] = u_hist   [:, di]

        feat_t = torch.tensor(feat_np, dtype=torch.float32, device=device)
        feat_scaled = (feat_t - x_mean) / x_std          # (B, input_size)

        # ---- Physics step (numpy, no grad) ----
        u_now = u_steps[:, h]
        phi_phys,  phid_phys = rk4_step_batch(phi_cur, phid_cur, u_now, Ts)

        phi_phys_t  = torch.tensor(phi_phys,  dtype=torch.float32, device=device)
        phid_phys_t = torch.tensor(phid_phys, dtype=torch.float32, device=device)

        # ---- NN residual  (B, 2), one forward pass for whole batch ----
        res_sc = model(feat_scaled)                        # (B, 2)
        res    = res_sc * y_std + y_mean                   # (B, 2) inv-scaled

        phi_next  = phi_phys_t  + res[:, 0]               # (B,)
        phid_next = phid_phys_t + res[:, 1]               # (B,)

        # ---- Loss for this step ----
        err_phi  = phi_next  - gt_phi_t [:, h]
        err_phid = phid_next - gt_phid_t[:, h]
        step_mse = (err_phi**2 + err_phid**2).mean()
        reg      = lambda_residual_reg * (res[:, 0]**2 + res[:, 1]**2).mean()

        total_loss = total_loss + step_mse + reg

        # ---- Advance state buffers (detach → numpy) ----
        phi_cur  = phi_next.detach().cpu().numpy()
        phid_cur = phid_next.detach().cpu().numpy()

        # Shift history: drop oldest tap, append current prediction
        # delays list is [0,1,2,5,10] — rebuild from raw signal where
        # we still have ground truth; use predicted state only for tap 0
        phi_hist  = np.stack([
            (phi_cur if dly == 0
             else PA_np [k0s + h + 1 - dly])
            for dly in delays
        ], axis=1)
        phid_hist = np.stack([
            (phid_cur if dly == 0
             else PAD_np[k0s + h + 1 - dly])
            for dly in delays
        ], axis=1)
        u_hist = np.stack([
            u_np[k0s + h + 1 - dly]
            for dly in delays
        ], axis=1)

    return total_loss / horizon


# ============================================================
# MODEL, LOSS, OPTIMISER
# ============================================================

model = ImprovedHybridNN(input_size).to(device)
print("\nModel Architecture:")
print(model)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=scheduler_factor,
    patience=scheduler_patience,
)


# ============================================================
# TRAINING LOOP  (one-step + curriculum rollout)
# ============================================================

best_val_loss      = np.inf
best_epoch         = -1
best_model_state   = None
early_stop_counter = 0
train_losses       = []
val_losses         = []

print("\nStarting training...\n")

for epoch in range(epochs):

    # Current rollout horizon (curriculum ramp)
    horizon = min(
        rollout_horizon_start + epoch // rollout_ramp_epochs,
        rollout_horizon_max
    )

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    model.train()
    train_loss_epoch = 0.0

    for X_batch, Y_batch, idx_batch in train_loader:
        optimizer.zero_grad()

        # --- One-step loss ---
        X_noisy  = X_batch + noise_std_train * torch.randn_like(X_batch)
        pred     = model(X_noisy)
        loss_1   = criterion(pred, Y_batch)

        # Residual regularisation on one-step output
        res_phys = pred * y_std + y_mean          # inverse-scale
        reg_loss = lambda_residual_reg * res_phys.pow(2).mean()

        # --- Multi-step rollout loss ---
        if horizon > 1:
            loss_roll = rollout_loss_fn(
                model, idx_batch,
                PA_train_t, PAD_train_t, u_train_t,
                horizon,
            )
        else:
            loss_roll = torch.zeros(1, device=device)

        loss = loss_1 + reg_loss + lambda_rollout * loss_roll

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        train_loss_epoch += loss_1.item() * X_batch.size(0)

    train_loss_epoch /= len(train_loader.dataset)

    # --------------------------------------------------
    # VALIDATE  (one-step only for speed)
    # --------------------------------------------------
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), Y_val_t).item()

    train_losses.append(train_loss_epoch)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    # --------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        best_epoch       = epoch
        best_model_state = copy.deepcopy(model.state_dict())
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | horizon {horizon:2d} | "
            f"Train: {train_loss_epoch:.8f} | Val: {val_loss:.8f}"
        )

    if early_stop_counter >= early_stop_patience:
        print("\nEarly stopping triggered.")
        break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(best_model_state)

model.eval()
with torch.no_grad():
    final_test_loss = criterion(model(X_test_t), Y_test_t).item()

print("\n===================================")
print("BEST MODEL LOADED")
print("===================================")
print(f"Best epoch     : {best_epoch}")
print(f"Best val loss  : {best_val_loss:.10f}")
print(f"Final test loss: {final_test_loss:.10f}")


# ============================================================
# SAVE ARTEFACTS
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "delays": cfg.delays,
        "Ts": Ts,
    },
    model_save_path,
)
joblib.dump(x_scaler, x_scaler_save_path)
joblib.dump(y_scaler, y_scaler_save_path)

print("\nModel and scalers saved.")

__all__ = [
    "model", "x_scaler", "y_scaler",
    "X_test_t", "Y_test_t",
    "train_losses", "val_losses",
]