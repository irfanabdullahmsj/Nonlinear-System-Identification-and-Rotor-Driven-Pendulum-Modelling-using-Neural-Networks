"""
dataset.py
==========
Load .mat experiment files, compute residual targets,
build delayed-feature vectors, and expose raw signals
for the rollout training loop.
"""

import os
import numpy as np
from scipy.io import loadmat

from config import data_dir, delays, max_delay
from physics import rk4_step


# ============================================================
# NOISE LEVELS  (sensor noise augmentation during training)
# ============================================================

_NOISE_PHI     = np.deg2rad(0.5)   # [rad]
_NOISE_PHI_DOT = np.deg2rad(2.0)   # [rad/s]


# ============================================================
# RAW SIGNAL LOADER  (needed by train.py rollout loss)
# ============================================================

def load_raw_signals(file_name: str):
    """
    Return (PA [rad], PAD [rad/s], u) as float32 numpy arrays,
    NaN-cleaned and length-aligned.
    """
    path = os.path.join(data_dir, file_name)
    mat  = loadmat(path)

    PA_deg  = mat["PA"].flatten().astype(np.float32)
    PAD_deg = mat["PAD"].flatten().astype(np.float32)
    u_data  = mat["u"].flatten().astype(np.float32)

    min_len = min(len(PA_deg), len(PAD_deg), len(u_data))
    PA_deg  = PA_deg[:min_len]
    PAD_deg = PAD_deg[:min_len]
    u_data  = u_data[:min_len]

    valid   = ~np.isnan(PA_deg) & ~np.isnan(PAD_deg) & ~np.isnan(u_data)
    PA_deg  = PA_deg[valid]
    PAD_deg = PAD_deg[valid]
    u_data  = u_data[valid]

    return np.deg2rad(PA_deg), np.deg2rad(PAD_deg), u_data


# ============================================================
# FEATURE / TARGET BUILDER
# ============================================================

def load_experiment_file(file_name: str, Ts: float):
    """
    Load a single .mat experiment and build (X, Y) arrays
    together with bookkeeping arrays for evaluation.

    Returns
    -------
    X                : (N, input_size)  float32  – delayed feature vectors
    Y                : (N, 2)           float32  – residual targets [Δphi, Δphi_dot]
    phi_true         : (N,)  true phi at k+1
    phi_dot_true     : (N,)  true phi_dot at k+1
    phi_phys         : (N,)  physics-only phi at k+1
    phi_dot_phys     : (N,)  physics-only phi_dot at k+1
    """

    PA, PAD, u_data = load_raw_signals(file_name)

    X_list            = []
    Y_list            = []
    phi_true_list     = []
    phi_dot_true_list = []
    phi_phys_list     = []
    phi_dot_phys_list = []

    for k in range(max_delay, len(PA) - 1):

        phi_k     = PA[k]
        phi_dot_k = PAD[k]
        u_k       = u_data[k]

        # sensor noise augmentation on current state
        phi_k_noisy     = phi_k     + np.random.normal(0, _NOISE_PHI)
        phi_dot_k_noisy = phi_dot_k + np.random.normal(0, _NOISE_PHI_DOT)

        phi_next_true     = PA[k + 1]
        phi_dot_next_true = PAD[k + 1]

        phi_next_phys, phi_dot_next_phys = rk4_step(
            phi_k_noisy, phi_dot_k_noisy, u_k, Ts
        )

        residual_phi     = phi_next_true     - phi_next_phys
        residual_phi_dot = phi_dot_next_true - phi_dot_next_phys

        # delayed input features (with noise augmentation)
        features = []
        for dly in delays:
            idx = k - dly
            features.extend([
                PA[idx]  + np.random.normal(0, _NOISE_PHI),
                PAD[idx] + np.random.normal(0, _NOISE_PHI_DOT),
                u_data[idx],
            ])

        X_list.append(features)
        Y_list.append([residual_phi, residual_phi_dot])

        phi_true_list.append(phi_next_true)
        phi_dot_true_list.append(phi_dot_next_true)
        phi_phys_list.append(phi_next_phys)
        phi_dot_phys_list.append(phi_dot_next_phys)

    return (
        np.array(X_list,           dtype=np.float32),
        np.array(Y_list,           dtype=np.float32),
        np.array(phi_true_list),
        np.array(phi_dot_true_list),
        np.array(phi_phys_list),
        np.array(phi_dot_phys_list),
    )


# ============================================================
# CONVENIENCE: LOAD ALL THREE SPLITS
# ============================================================

def load_all_splits(train_file, val_file, test_file, Ts):
    print("\nLoading training data...")
    train = load_experiment_file(train_file, Ts)
    print(f"  Training samples   : {len(train[0])}")

    print("Loading validation data...")
    val = load_experiment_file(val_file, Ts)
    print(f"  Validation samples : {len(val[0])}")

    print("Loading test data...")
    test = load_experiment_file(test_file, Ts)
    print(f"  Test samples       : {len(test[0])}")

    return {"train": train, "val": val, "test": test}