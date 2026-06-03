import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import joblib

from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")
print("=" * 50)
print("DEVICE CONFIGURATION")
print("=" * 50)

print("Using device:", device)


# ============================================================
# FILE PATHS
# ============================================================

data_dir = r"I:\h_da\Summer 2026\Team_Project\Datasets\NODE"

train_file = "Reduced_Exp_12May_1457_upsamp_filt.mat"

val_file = "Reduced_Exp_12May_1500_upsamp_filt.mat"

test_file = "Reduced_Exp_12May_1503_upsamp_filt.mat"

Ts = 0.002


# ============================================================
# IMPROVED DELAY CONFIGURATION
# ============================================================

delays = [0, 1, 2, 5, 10]

max_delay = max(delays)

input_size = len(delays) * 3

print("\nDelay configuration:", delays)
print("Maximum delay:", max_delay)
print("Input size:", input_size)


# ============================================================
# PHYSICAL MODEL PARAMETERS
# ============================================================

g = 9.81

mb = 0.3
mr = 0.087
ms = 0.430

mbrs = mb + mr + ms

lb = 0.473
lr = 0.5

x = 0.09

lofXgeom = (
    (x * ms + lb / 2 * mb + lr * mr)
    / mbrs
)

JofXgeom = (
    1 / 3 * mb * lb**2
    + mr * lr**2
    + ms * x**2
)

d = 0.0061

a0 = g * mbrs * lofXgeom
a1 = d
a2 = JofXgeom

print("\nPhysical parameters:")
print("a0 =", a0)
print("a1 =", a1)
print("a2 =", a2)


# ============================================================
# MOTOR MODEL
# ============================================================

def Mr_cubic(u):

    return (
        -0.00000992 * u**3
        + 0.00105529 * u**2
        - 0.00489381 * u
    )



# ============================================================
# RK4 DYNAMICS
# ============================================================

def dynamics(phi, phi_dot, u):
    phi_ddot = (
        -a1 * phi_dot
        -a0 * np.sin(phi)
        + Mr_cubic(u)
    ) / a2

    return phi_dot, phi_ddot


def rk4_step(phi, phi_dot, u, Ts):

    k1_phi, k1_dot = dynamics(
        phi,
        phi_dot,
        u
    )

    k2_phi, k2_dot = dynamics(
        phi + 0.5 * Ts * k1_phi,
        phi_dot + 0.5 * Ts * k1_dot,
        u
    )

    k3_phi, k3_dot = dynamics(
        phi + 0.5 * Ts * k2_phi,
        phi_dot + 0.5 * Ts * k2_dot,
        u
    )

    k4_phi, k4_dot = dynamics(
        phi + Ts * k3_phi,
        phi_dot + Ts * k3_dot,
        u
    )

    phi_next = phi + (Ts / 6.0) * (
        k1_phi
        + 2 * k2_phi
        + 2 * k3_phi
        + k4_phi
    )

    phi_dot_next = phi_dot + (Ts / 6.0) * (
        k1_dot
        + 2 * k2_dot
        + 2 * k3_dot
        + k4_dot
    )

    return phi_next, phi_dot_next


# ============================================================
# LOAD DATASET
# ============================================================

def load_experiment_file(file_name):

    path = os.path.join(data_dir, file_name)

    mat = loadmat(path)

    PA_deg = mat["PA"].flatten().astype(np.float32)
    PAD_deg = mat["PAD"].flatten().astype(np.float32)
    u_data = mat["u"].flatten().astype(np.float32)

    min_len = min(
        len(PA_deg),
        len(PAD_deg),
        len(u_data)
    )

    PA_deg = PA_deg[:min_len]
    PAD_deg = PAD_deg[:min_len]
    u_data = u_data[:min_len]

    valid_idx = (
        ~np.isnan(PA_deg)
        & ~np.isnan(PAD_deg)
        & ~np.isnan(u_data)
    )

    PA_deg = PA_deg[valid_idx]
    PAD_deg = PAD_deg[valid_idx]
    u_data = u_data[valid_idx]

    PA = np.deg2rad(PA_deg)
    PAD = np.deg2rad(PAD_deg)

    X_list = []
    Y_list = []

    phi_true_used = []
    phi_dot_true_used = []

    phi_phys_used = []
    phi_dot_phys_used = []

    for k in range(max_delay, len(PA) - 1):

        phi_k = PA[k]
        phi_dot_k = PAD[k]

        # simulate recursive drift
        phi_k_noisy = (
            phi_k
            + np.random.normal(0, np.deg2rad(0.5))
        )

        phi_dot_k_noisy = (
            phi_dot_k
            + np.random.normal(0, np.deg2rad(2.0))
        )
        u_k = u_data[k]

        # ---------------------------------------
        # TRUE NEXT STATE
        # ---------------------------------------

        phi_next_true = PA[k + 1]
        phi_dot_next_true = PAD[k + 1]

        # ---------------------------------------
        # RK4 PHYSICAL MODEL
        # ---------------------------------------

        phi_next_phys, phi_dot_next_phys = rk4_step(
            phi_k_noisy,
            phi_dot_k_noisy,
            u_k,
            Ts
        )

        # ---------------------------------------
        # RESIDUAL TARGETS
        # ---------------------------------------

        residual_phi = (
            phi_next_true
            - phi_next_phys
        )

        residual_phi_dot = (
            phi_dot_next_true
            - phi_dot_next_phys
        )

        # ---------------------------------------
        # DELAYED FEATURES
        # ---------------------------------------

        features = []

        for dly in delays:

            idx = k - dly

            features.extend([
                PA[idx] + np.random.normal(
                    0,
                    np.deg2rad(0.5)
                ),

                PAD[idx] + np.random.normal(
                    0,
                    np.deg2rad(2.0)
                ),

                u_data[idx]
            ])

        X_list.append(features)

        Y_list.append([
            residual_phi,
            residual_phi_dot
        ])

        phi_true_used.append(phi_next_true)
        phi_dot_true_used.append(phi_dot_next_true)

        phi_phys_used.append(phi_next_phys)
        phi_dot_phys_used.append(phi_dot_next_phys)

    X = np.array(X_list, dtype=np.float32)

    Y = np.array(Y_list, dtype=np.float32)

    phi_true_used = np.array(phi_true_used)
    phi_dot_true_used = np.array(phi_dot_true_used)

    phi_phys_used = np.array(phi_phys_used)
    phi_dot_phys_used = np.array(phi_dot_phys_used)

    return (
        X,
        Y,
        phi_true_used,
        phi_dot_true_used,
        phi_phys_used,
        phi_dot_phys_used
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

(
    X_train,
    Y_train,
    _,
    _,
    _,
    _
) = load_experiment_file(train_file)

print("Training samples:", len(X_train))
# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

(
    X_val,
    Y_val,
    _,
    _,
    _,
    _
) = load_experiment_file(val_file)

print("Validation samples:", len(X_val))

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

(
    X_test,
    Y_test,
    phi_true,
    phi_dot_true,
    phi_phys,
    phi_dot_phys
) = load_experiment_file(test_file)

print("Test samples:", len(X_test))


# ============================================================
# NORMALIZATION
# ============================================================

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
Y_train_scaled = y_scaler.fit_transform(Y_train)

X_val_scaled = x_scaler.transform(X_val)
Y_val_scaled = y_scaler.transform(Y_val)

X_test_scaled = x_scaler.transform(X_test)
Y_test_scaled = y_scaler.transform(Y_test)

X_train_t = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
).to(device)

Y_train_t = torch.tensor(
    Y_train_scaled,
    dtype=torch.float32
).to(device)

X_val_t = torch.tensor(
    X_val_scaled,
    dtype=torch.float32
).to(device)

Y_val_t = torch.tensor(
    Y_val_scaled,
    dtype=torch.float32
).to(device)

X_test_t = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
).to(device)

Y_test_t = torch.tensor(
    Y_test_scaled,
    dtype=torch.float32
).to(device)


# ============================================================
# TRAIN LOADER
# ============================================================

train_loader = DataLoader(
    TensorDataset(X_train_t, Y_train_t),
    batch_size=64,
    shuffle=True
)


# ============================================================
# IMPROVED NEURAL NETWORK
# ============================================================
class ImprovedHybridNN(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_size, 64),
            nn.SiLU(),

            nn.Linear(64, 64),
            nn.SiLU(),

            nn.Linear(64, 32),
            nn.SiLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):

        return self.net(x)


model = ImprovedHybridNN(
    input_size=input_size
).to(device)

print("\nModel Architecture:")
print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=20
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

epochs = 1000

best_loss = np.inf
best_epoch = -1

best_model_state = None

early_stop_patience = 80
early_stop_counter = 0

train_losses = []
val_losses = []


# ============================================================
# TRAINING LOOP
# ============================================================

print("\nStarting training...\n")

for epoch in range(epochs):

    # =====================================================
    # TRAINING
    # =====================================================

    model.train()

    train_loss_epoch = 0.0

    for X_batch, Y_batch in train_loader:

        optimizer.zero_grad()

        noise_std = 0.01

        X_noisy = (
        X_batch
        + noise_std * torch.randn_like(X_batch)
        )

        pred = model(X_noisy)

        loss = criterion(
        pred,
        Y_batch
        )       

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_loss_epoch += (
            loss.item()
            * X_batch.size(0)
        )

    train_loss_epoch /= len(train_loader.dataset)

    # =====================================================
    # VALIDATION EVALUATION
    # =====================================================

    model.eval()

    with torch.no_grad():

        pred_val = model(X_val_t)

        val_loss = criterion(
            pred_val,
            Y_val_t
        ).item()

    train_losses.append(train_loss_epoch)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if val_loss < best_loss:

        best_loss = val_loss

        best_epoch = epoch

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        early_stop_counter = 0

    else:

        early_stop_counter += 1

    # =====================================================
    # PRINT
    # =====================================================

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch:4d} | "
            f"Train Loss: {train_loss_epoch:.10f} | "
            f"Val Loss: {val_loss:.10f}"
        )

    # =====================================================
    # EARLY STOPPING
    # =====================================================

    if early_stop_counter >= early_stop_patience:

        print("\nEarly stopping triggered.")
        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(best_model_state)
# =====================================================
# FINAL TEST EVALUATION
# =====================================================

model.eval()

with torch.no_grad():

    pred_test = model(X_test_t)

    final_test_loss = criterion(
        pred_test,
        Y_test_t
    ).item()

print("\nFinal Test Loss:", final_test_loss)

print("\n===================================")
print("BEST MODEL LOADED")
print("===================================")

print("Best epoch:", best_epoch)
print("Best test loss:", best_loss)


# ============================================================
# HYBRID PREDICTION
# ============================================================

model.eval()

with torch.no_grad():

    residual_scaled = model(X_test_t).cpu().numpy()

residual_pred = y_scaler.inverse_transform(
    residual_scaled
)

phi_res = residual_pred[:, 0]
phi_dot_res = residual_pred[:, 1]

# ------------------------------------------------------------
# FINAL HYBRID PREDICTION
# ------------------------------------------------------------

phi_hybrid = phi_phys + phi_res

phi_dot_hybrid = (
    phi_dot_phys
    + phi_dot_res
)

# ============================================================
# CONVERT TO DEGREES
# ============================================================

phi_true_deg = np.rad2deg(phi_true)

phi_phys_deg = np.rad2deg(phi_phys)

phi_hybrid_deg = np.rad2deg(phi_hybrid)

phi_dot_true_deg = np.rad2deg(
    phi_dot_true
)

phi_dot_phys_deg = np.rad2deg(
    phi_dot_phys
)

phi_dot_hybrid_deg = np.rad2deg(
    phi_dot_hybrid
)


# ============================================================
# METRICS
# ============================================================

mse_phys = np.mean(
    (phi_true_deg - phi_phys_deg)**2
)

mse_hybrid = np.mean(
    (phi_true_deg - phi_hybrid_deg)**2
)

rmse_phys = np.sqrt(mse_phys)

rmse_hybrid = np.sqrt(mse_hybrid)

print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print(f"Physical RMSE : {rmse_phys:.6f} deg")

print(f"Hybrid RMSE   : {rmse_hybrid:.6f} deg")

print(
    f"Improvement   : "
    f"{rmse_phys - rmse_hybrid:.6f} deg"
)


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "delays": delays,
        "Ts": Ts,
    },
    "improved_hybrid_model.pth"
)

joblib.dump(
    x_scaler,
    "x_scaler.pkl"
)

joblib.dump(
    y_scaler,
    "y_scaler.pkl"
)

print("\nModel saved successfully.")

model.eval()

with torch.no_grad():

    residual_scaled = model(X_test_t).cpu().numpy()

residual_pred = y_scaler.inverse_transform(
    residual_scaled
)

phi_res = residual_pred[:, 0]
phi_dot_res = residual_pred[:, 1]

# ------------------------------------------------------------
# ONE STEP HYBRID PREDICTION
# ------------------------------------------------------------

phi_hybrid = phi_phys + phi_res

phi_dot_hybrid = (
    phi_dot_phys
    + phi_dot_res
)

# ============================================================
# CONVERT TO DEGREES
# ============================================================

phi_true_deg = np.rad2deg(phi_true)

phi_hybrid_deg = np.rad2deg(phi_hybrid)

phi_dot_true_deg = np.rad2deg(
    phi_dot_true
)

phi_dot_hybrid_deg = np.rad2deg(
    phi_dot_hybrid
)

# ============================================================
# TIME VECTOR
# ============================================================

t = np.arange(len(phi_true_deg)) * Ts

# ============================================================
# ONE STEP PREDICTION PLOTS
# ============================================================

plt.figure(figsize=(14, 8))

# ------------------------------------------------------------
# PA
# ------------------------------------------------------------

plt.subplot(2, 1, 1)

plt.plot(
    t,
    phi_true_deg,
    label="True PA"
)

plt.plot(
    t,
    phi_hybrid_deg,
    "--",
    label="Predicted PA"
)

plt.ylabel("PA [deg]")

plt.title("One-Step Prediction")

plt.grid(True)

plt.legend()

# ------------------------------------------------------------
# PAD
# ------------------------------------------------------------

plt.subplot(2, 1, 2)

plt.plot(
    t,
    phi_dot_true_deg,
    label="True PAD"
)

plt.plot(
    t,
    phi_dot_hybrid_deg,
    "--",
    label="Predicted PAD"
)

plt.xlabel("Time [s]")

plt.ylabel("PAD [deg/s]")

plt.grid(True)

plt.legend()

plt.tight_layout()

# ============================================================
# RECURSIVE HYBRID SIMULATION
# ============================================================

print("\nRunning recursive simulation...")

N = len(phi_true)

# ------------------------------------------------------------
# INITIAL CONDITIONS
# ------------------------------------------------------------

phi_sim = np.zeros(N)
phi_dot_sim = np.zeros(N)

phi_sim[:max_delay] = phi_true[:max_delay]
phi_dot_sim[:max_delay] = phi_dot_true[:max_delay]

# ------------------------------------------------------------
# PURE PHYSICAL SIMULATION
# ------------------------------------------------------------

phi_phys_sim = np.zeros(N)
phi_dot_phys_sim = np.zeros(N)

phi_phys_sim[:max_delay] = phi_true[:max_delay]
phi_dot_phys_sim[:max_delay] = phi_dot_true[:max_delay]

# ------------------------------------------------------------
# INPUT SIGNAL
# ------------------------------------------------------------

path = os.path.join(data_dir, test_file)

mat = loadmat(path)

u_full = mat["u"].flatten().astype(np.float32)

u_full = u_full[:N + max_delay]

# ============================================================
# STABILIZATION FACTORS
# ============================================================

alpha_phi = 1.0
alpha_pad = 1.0
phi_clip_count = 0
pad_clip_count = 0
# ============================================================
# RECURSIVE LOOP
# ============================================================

for k in range(max_delay, N - 1):

    # ========================================================
    # PURE PHYSICAL MODEL
    # ========================================================

    phi_phys_next, phi_dot_phys_next = rk4_step(
        phi_phys_sim[k],
        phi_dot_phys_sim[k],
        u_full[k],
        Ts
    )

    phi_phys_sim[k + 1] = phi_phys_next
    phi_dot_phys_sim[k + 1] = phi_dot_phys_next

    # ========================================================
    # BUILD FEATURES
    # ========================================================

    features = []

    for dly in delays:

        idx = k - dly

        features.extend([
            phi_sim[idx],
            phi_dot_sim[idx],
            u_full[idx]
        ])

    features = np.array(
        features,
        dtype=np.float32
    ).reshape(1, -1)

    # ========================================================
    # SCALE FEATURES
    # ========================================================

    features_scaled = x_scaler.transform(
        features
    )

    X_tensor = torch.tensor(
        features_scaled,
        dtype=torch.float32
    ).to(device)

    # ========================================================
    # PHYSICAL MODEL STEP
    # ========================================================

    phi_phys_next, phi_dot_phys_next = rk4_step(
        phi_sim[k],
        phi_dot_sim[k],
        u_full[k],
        Ts
    )

    # ========================================================
    # NN RESIDUAL
    # ========================================================

    with torch.no_grad():

        residual_scaled = model(
            X_tensor
        ).cpu().numpy()

    residual = y_scaler.inverse_transform(
        residual_scaled
    )[0]

    phi_residual = residual[0]
    phi_dot_residual = residual[1]

    # ========================================================
    # STABILIZED HYBRID UPDATE
    # ========================================================

    phi_sim[k + 1] = (
        phi_phys_next
        + alpha_phi * phi_residual
    )

    phi_dot_sim[k + 1] = (
        phi_dot_phys_next
        + alpha_pad * phi_dot_residual
    )

    # ========================================================
# STABILITY CLIPPING
# ========================================================

    if phi_sim[k + 1] > np.deg2rad(90):
        phi_clip_count += 1
        phi_sim[k + 1] = np.deg2rad(90)

    elif phi_sim[k + 1] < np.deg2rad(-90):
        phi_clip_count += 1
        phi_sim[k + 1] = np.deg2rad(-90)


    if phi_dot_sim[k + 1] > np.deg2rad(400):
        pad_clip_count += 1
        phi_dot_sim[k + 1] = np.deg2rad(400)

    elif phi_dot_sim[k + 1] < np.deg2rad(-400):
        pad_clip_count += 1
        phi_dot_sim[k + 1] = np.deg2rad(-400)

# ============================================================
# CONVERT TO DEGREES
# ============================================================

phi_sim_deg = np.rad2deg(phi_sim)

phi_phys_sim_deg = np.rad2deg(phi_phys_sim)

phi_dot_sim_deg = np.rad2deg(phi_dot_sim)

phi_dot_phys_sim_deg = np.rad2deg(phi_dot_phys_sim)

# ============================================================
# RECURSIVE RMSE
# ============================================================

rmse_phys_recursive = np.sqrt(
    np.mean(
        (phi_true_deg - phi_phys_sim_deg)**2
    )
)

rmse_hybrid_recursive = np.sqrt(
    np.mean(
        (phi_true_deg - phi_sim_deg)**2
    )
)

print("\n===================================")
print("RECURSIVE SIMULATION RESULTS")
print("===================================")

print(
    f"Physics Recursive RMSE : "
    f"{rmse_phys_recursive:.6f} deg"
)

print(
    f"Hybrid Recursive RMSE  : "
    f"{rmse_hybrid_recursive:.6f} deg"
)

print(
    f"Improvement            : "
    f"{rmse_phys_recursive - rmse_hybrid_recursive:.6f} deg"
)
print("\n===================================")
print("CLIPPING STATISTICS")
print("===================================")

print(f"PA clips  : {phi_clip_count}")
print(f"PAD clips : {pad_clip_count}")
# ============================================================
# RECURSIVE RMSE COMPARISON
# ============================================================

rmse_phys_recursive = np.sqrt(
    np.mean(
        (phi_true_deg - phi_phys_sim_deg) ** 2
    )
)

rmse_hybrid_recursive = np.sqrt(
    np.mean(
        (phi_true_deg - phi_sim_deg) ** 2
    )
)

print("\n===================================")
print("RECURSIVE SIMULATION RESULTS")
print("===================================")

print(
    f"Physics Recursive RMSE : "
    f"{rmse_phys_recursive:.6f} deg"
)

print(
    f"Hybrid Recursive RMSE  : "
    f"{rmse_hybrid_recursive:.6f} deg"
)

improvement = (
    rmse_phys_recursive
    - rmse_hybrid_recursive
)

print(
    f"Improvement            : "
    f"{improvement:.6f} deg"
)

if rmse_hybrid_recursive < rmse_phys_recursive:
    print("\n✓ Hybrid model is better.")
else:
    print("\n✗ Physics model is better.")
# ============================================================
# RECURSIVE PLOTS
# ============================================================

plt.figure(figsize=(14, 8))

# ------------------------------------------------------------
# PA
# ------------------------------------------------------------

plt.subplot(2, 1, 1)

plt.plot(
    t,
    phi_true_deg,
    label="True PA"
)

plt.plot(
    t,
    phi_phys_sim_deg,
    "--",
    label="Physics PA"
)

plt.plot(
    t,
    phi_sim_deg,
    "--",
    label="Hybrid Recursive PA"
)

plt.ylabel("PA [deg]")

plt.title("Recursive Simulation")

plt.grid(True)

plt.legend()

# ------------------------------------------------------------
# PAD
# ------------------------------------------------------------

plt.subplot(2, 1, 2)

plt.plot(
    t,
    phi_dot_true_deg,
    label="True PAD"
)

plt.plot(
    t,
    phi_dot_phys_sim_deg,
    "--",
    label="Physics PAD"
)

plt.plot(
    t,
    phi_dot_sim_deg,
    "--",
    label="Hybrid Recursive PAD"
)

plt.xlabel("Time [s]")

plt.ylabel("PAD [deg/s]")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()