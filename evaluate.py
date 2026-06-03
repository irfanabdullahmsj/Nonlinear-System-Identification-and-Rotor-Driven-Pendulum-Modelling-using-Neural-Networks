"""
evaluate.py
===========
One-step prediction metrics, recursive closed-loop simulation,
RMSE comparison, and all diagnostic plots.

Run after train.py (or import the trained artefacts from there).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat

import config as cfg
from config import (
    data_dir, test_file, Ts, max_delay, delays, device,
    phi_clip_deg, phi_dot_clip_deg,
    alpha_phi, alpha_pad,
)
from physics import rk4_step


# ============================================================
# LOAD TRAINED ARTEFACTS
# ============================================================
# Import the trained model, scalers, and test tensors that
# were produced (and saved in memory) by train.py.

from train import (
    model, x_scaler, y_scaler,
    X_test_t, Y_test_t,
)

# Ground-truth / physics arrays produced during data loading
from dataset import load_experiment_file

_, _, phi_true, phi_dot_true, phi_phys, phi_dot_phys = \
    load_experiment_file(test_file, Ts)


# ============================================================
# ONE-STEP RESIDUAL PREDICTION
# ============================================================

model.eval()
with torch.no_grad():
    residual_scaled = model(X_test_t).cpu().numpy()

residual_pred = y_scaler.inverse_transform(residual_scaled)
phi_res     = residual_pred[:, 0]
phi_dot_res = residual_pred[:, 1]

phi_hybrid     = phi_phys     + phi_res
phi_dot_hybrid = phi_dot_phys + phi_dot_res

# ---- degrees -----------------------------------------------
phi_true_deg       = np.rad2deg(phi_true)
phi_phys_deg       = np.rad2deg(phi_phys)
phi_hybrid_deg     = np.rad2deg(phi_hybrid)
phi_dot_true_deg   = np.rad2deg(phi_dot_true)
phi_dot_phys_deg   = np.rad2deg(phi_dot_phys)
phi_dot_hybrid_deg = np.rad2deg(phi_dot_hybrid)

# ---- metrics ------------------------------------------------
rmse_phys   = np.sqrt(np.mean((phi_true_deg - phi_phys_deg)**2))
rmse_hybrid = np.sqrt(np.mean((phi_true_deg - phi_hybrid_deg)**2))

print("\n===================================")
print("ONE-STEP RESULTS")
print("===================================")
print(f"Physical RMSE : {rmse_phys:.6f} deg")
print(f"Hybrid RMSE   : {rmse_hybrid:.6f} deg")
print(f"Improvement   : {rmse_phys - rmse_hybrid:.6f} deg")


# ============================================================
# ONE-STEP PLOTS
# ============================================================

t = np.arange(len(phi_true_deg)) * Ts

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(t, phi_true_deg,   label="True PA")
plt.plot(t, phi_hybrid_deg, "--", label="Predicted PA")
plt.ylabel("PA [deg]")
plt.title("One-Step Prediction")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, phi_dot_true_deg,   label="True PAD")
plt.plot(t, phi_dot_hybrid_deg, "--", label="Predicted PAD")
plt.xlabel("Time [s]")
plt.ylabel("PAD [deg/s]")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# # ============================================================
# # RECURSIVE SIMULATION
# # ============================================================

# print("\nRunning recursive simulation...")

# # load full input signal
# mat    = loadmat(os.path.join(data_dir, test_file))
# u_full = mat["u"].flatten().astype(np.float32)

# N = len(phi_true)
# u_full = u_full[:N + max_delay]

# # initialise state buffers
# phi_sim     = np.zeros(N)
# phi_dot_sim = np.zeros(N)
# phi_sim[:max_delay]     = phi_true[:max_delay]
# phi_dot_sim[:max_delay] = phi_dot_true[:max_delay]

# phi_phys_sim     = np.zeros(N)
# phi_dot_phys_sim = np.zeros(N)
# phi_phys_sim[:max_delay]     = phi_true[:max_delay]
# phi_dot_phys_sim[:max_delay] = phi_dot_true[:max_delay]

# # clip counters
# phi_clip_count = 0
# pad_clip_count = 0

# phi_clip_rad     = np.deg2rad(phi_clip_deg)
# phi_dot_clip_rad = np.deg2rad(phi_dot_clip_deg)

# for k in range(max_delay, N - 1):

#     # ------ pure physics ----------------------------------------
#     phi_phys_next, phi_dot_phys_next = rk4_step(
#         phi_phys_sim[k], phi_dot_phys_sim[k], u_full[k], Ts
#     )
#     phi_phys_sim[k + 1]     = phi_phys_next
#     phi_dot_phys_sim[k + 1] = phi_dot_phys_next

#     # ------ build NN input features -----------------------------
#     features = []
#     for dly in delays:
#         idx = k - dly
#         features.extend([phi_sim[idx], phi_dot_sim[idx], u_full[idx]])

#     features_scaled = x_scaler.transform(
#         np.array(features, dtype=np.float32).reshape(1, -1)
#     )
#     X_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

#     # ------ physics step for hybrid state -----------------------
#     phi_p_next, phi_dot_p_next = rk4_step(
#         phi_sim[k], phi_dot_sim[k], u_full[k], Ts
#     )

#     # ------ NN residual -----------------------------------------
#     with torch.no_grad():
#         res_scaled = model(X_tensor).cpu().numpy()

#     residual = y_scaler.inverse_transform(res_scaled)[0]

#     # ------ hybrid update ---------------------------------------
#     phi_sim[k + 1]     = phi_p_next     + alpha_phi * residual[0]
#     phi_dot_sim[k + 1] = phi_dot_p_next + alpha_pad * residual[1]

#     # ------ stability clipping ----------------------------------
#     if   phi_sim[k + 1] >  phi_clip_rad:
#         phi_sim[k + 1] =  phi_clip_rad;  phi_clip_count += 1
#     elif phi_sim[k + 1] < -phi_clip_rad:
#         phi_sim[k + 1] = -phi_clip_rad;  phi_clip_count += 1

#     if   phi_dot_sim[k + 1] >  phi_dot_clip_rad:
#         phi_dot_sim[k + 1] =  phi_dot_clip_rad;  pad_clip_count += 1
#     elif phi_dot_sim[k + 1] < -phi_dot_clip_rad:
#         phi_dot_sim[k + 1] = -phi_dot_clip_rad;  pad_clip_count += 1


# # ---- degrees -----------------------------------------------
# phi_sim_deg          = np.rad2deg(phi_sim)
# phi_phys_sim_deg     = np.rad2deg(phi_phys_sim)
# phi_dot_sim_deg      = np.rad2deg(phi_dot_sim)
# phi_dot_phys_sim_deg = np.rad2deg(phi_dot_phys_sim)

# # ---- recursive metrics -------------------------------------
# rmse_phys_rec   = np.sqrt(np.mean((phi_true_deg - phi_phys_sim_deg)**2))
# rmse_hybrid_rec = np.sqrt(np.mean((phi_true_deg - phi_sim_deg)**2))

# print("\n===================================")
# print("RECURSIVE SIMULATION RESULTS")
# print("===================================")
# print(f"Physics Recursive RMSE : {rmse_phys_rec:.6f} deg")
# print(f"Hybrid Recursive RMSE  : {rmse_hybrid_rec:.6f} deg")
# print(f"Improvement            : {rmse_phys_rec - rmse_hybrid_rec:.6f} deg")
# print("\n===================================")
# print("CLIPPING STATISTICS")
# print("===================================")
# print(f"PA clips  : {phi_clip_count}")
# print(f"PAD clips : {pad_clip_count}")


# # ============================================================
# # RECURSIVE PLOTS
# # ============================================================

# plt.figure(figsize=(14, 8))

# plt.subplot(2, 1, 1)
# plt.plot(t, phi_true_deg,      label="True PA")
# plt.plot(t, phi_phys_sim_deg,  "--", label="Physics PA")
# plt.plot(t, phi_sim_deg,       "--", label="Hybrid Recursive PA")
# plt.ylabel("PA [deg]")
# plt.title("Recursive Simulation")
# plt.grid(True)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(t, phi_dot_true_deg,      label="True PAD")
# plt.plot(t, phi_dot_phys_sim_deg,  "--", label="Physics PAD")
# plt.plot(t, phi_dot_sim_deg,       "--", label="Hybrid Recursive PAD")
# plt.xlabel("Time [s]")
# plt.ylabel("PAD [deg/s]")
# plt.grid(True)
# plt.legend()

# plt.tight_layout()
# plt.show()