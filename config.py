"""
config.py
=========
Central configuration: file paths, sampling time,
delay settings, physical model parameters.
"""

import os
import numpy as np

# ============================================================
# DEVICE
# ============================================================

import torch
device = torch.device("cpu")

# ============================================================
# FILE PATHS
# ============================================================

data_dir = r"I:\h_da\Summer 2026\Team_Project\Datasets\NODE"

train_file = "Exp_12May_1457_upsamp_filt.mat"
val_file   = "Exp_12May_1500_upsamp_filt.mat"
test_file  = "Exp_12May_1503_upsamp_filt.mat"

Ts = 0.002  # sampling period [s]

# ============================================================
# DELAY CONFIGURATION
# ============================================================

delays     = [0, 1, 2, 5, 10]
max_delay  = max(delays)
input_size = len(delays) * 3  # (phi, phi_dot, u) per tap

# ============================================================
# PHYSICAL MODEL PARAMETERS
# ============================================================

g    = 9.81
mb   = 0.3
mr   = 0.087
ms   = 0.430
mbrs = mb + mr + ms

lb   = 0.473
lr   = 0.5
x    = 0.09

lofXgeom = (x * ms + lb / 2 * mb + lr * mr) / mbrs
JofXgeom = (1 / 3 * mb * lb**2 + mr * lr**2 + ms * x**2)

d  = 0.0061
a0 = g * mbrs * lofXgeom
a1 = d
a2 = JofXgeom

# ============================================================
# TRAINING HYPER-PARAMETERS
# ============================================================

batch_size          = 64
learning_rate       = 3e-4
weight_decay        = 1e-5
epochs              = 1000
early_stop_patience = 80
scheduler_patience  = 20
scheduler_factor    = 0.5
noise_std_train     = 0.01
grad_clip_norm      = 1.0

# ============================================================
# MULTI-STEP ROLLOUT TRAINING
# ============================================================
# Curriculum: ramp horizon from rollout_horizon_start up to
# rollout_horizon_max, increasing by 1 every rollout_ramp_epochs.
# One-step MSE loss weight vs multi-step rollout loss weight.

rollout_horizon_start = 1    # start with pure one-step loss
rollout_horizon_max   = 15   # maximum prediction horizon [steps]
rollout_ramp_epochs   = 50   # increase horizon every N epochs
lambda_rollout        = 0.5  # weight on multi-step loss term
lambda_residual_reg   = 0.05 # penalty on residual magnitude

# ============================================================
# SIMULATION / MPC STABILITY SETTINGS
# ============================================================

phi_clip_deg     =  90.0   # hard clip [deg]
phi_dot_clip_deg = 400.0   # hard clip [deg/s]

# Residual blending: 0 = pure physics, 1 = full hybrid
# Start conservative; increase only after verifying stability
alpha_phi = 1.0
alpha_pad = 1.0

# OOD (out-of-distribution) guard: fall back to pure physics
# if any scaled input feature exceeds this z-score threshold
ood_zscore_threshold = 3.0

# MPC prediction horizon [steps]  (used in mpc.py)
mpc_horizon = 15   # = 30 ms at Ts=0.002

# ============================================================
# SAVE PATHS
# ============================================================

model_save_path    = "improved_hybrid_model.pth"
x_scaler_save_path = "x_scaler.pkl"
y_scaler_save_path = "y_scaler.pkl"


# ============================================================
# QUICK DIAGNOSTICS
# ============================================================

if __name__ == "__main__":
    print("Delay configuration :", delays)
    print("Maximum delay        :", max_delay)
    print("Input size           :", input_size)
    print("\nPhysical parameters:")
    print(f"  a0 = {a0:.6f}")
    print(f"  a1 = {a1:.6f}")
    print(f"  a2 = {a2:.6f}")
    print("\nRollout training:")
    print(f"  horizon {rollout_horizon_start} → {rollout_horizon_max} "
          f"(+1 every {rollout_ramp_epochs} epochs)")
    print(f"  lambda_rollout      = {lambda_rollout}")
    print(f"  lambda_residual_reg = {lambda_residual_reg}")
    print(f"\nSimulation:")
    print(f"  alpha_phi / alpha_pad = {alpha_phi} / {alpha_pad}")
    print(f"  OOD threshold z = {ood_zscore_threshold}")
    print(f"  MPC horizon = {mpc_horizon} steps")