import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time

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

train_data = loadmat(
    r"C:\Users\CRA\Desktop\TeamProjectIrfan\Dataset\Exp_12May_1457_upsamp_filt.mat"
)

val_data = loadmat(
    r"C:\Users\CRA\Desktop\TeamProjectIrfan\Dataset\Exp_12May_1500_upsamp_filt.mat"
)

test_data = loadmat(
    r"C:\Users\CRA\Desktop\TeamProjectIrfan\Dataset\Exp_12May_1503_upsamp_filt.mat"
)

train_phi  = train_data['PA'].squeeze()
train_phid = train_data['PAD'].squeeze()
train_u    = train_data['u'].squeeze()
train_t    = train_data['t'].squeeze()
train_x    = train_data['x'].squeeze()

val_phi  = val_data['PA'].squeeze()
val_phid = val_data['PAD'].squeeze()
val_u    = val_data['u'].squeeze()
val_t    = val_data['t'].squeeze()
val_x    = val_data['x'].squeeze()

test_phi  = test_data['PA'].squeeze()
test_phid = test_data['PAD'].squeeze()
test_u    = test_data['u'].squeeze()
test_t    = test_data['t'].squeeze()
test_x    = test_data['x'].squeeze()

# ============================================================
# STATES
# ============================================================

train_states = np.column_stack((train_phi, train_phid))

val_states = np.column_stack((val_phi, val_phid))

test_states = np.column_stack((test_phi, test_phid))

# ============================================================
# NORMALIZATION
# ============================================================

state_scaler = StandardScaler()

train_states_scaled = state_scaler.fit_transform(
    train_states
)

val_states_scaled = state_scaler.fit_transform(
    val_states
)

test_states_scaled = state_scaler.fit_transform(
    test_states
)

u_scaler = StandardScaler()

train_u_scaled = u_scaler.fit_transform(
    train_u.reshape(-1, 1),
    train_x.reshape(-1, 1)
).flatten()

val_u_scaled = u_scaler.fit_transform(
    val_u.reshape(-1, 1),
    val_x.reshape(-1, 1)
).flatten()

test_u_scaled = u_scaler.fit_transform(
    test_u.reshape(-1, 1),
    test_x.reshape(-1, 1)
).flatten()

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

train_states = train_states_scaled
val_states   = val_states_scaled
test_states  = test_states_scaled

train_u = train_u_scaled
val_u   = val_u_scaled
test_u  = test_u_scaled

# ============================================================
# DATASET
# ============================================================

SEQ_LEN = 20

class TimeSeriesDataset(Dataset):

    def __init__(self, states, controls):

        self.states = states
        self.controls = controls

    def __len__(self):

        return len(self.states) - SEQ_LEN

    def __getitem__(self, idx):

        x0 = self.states[idx]

        target = self.states[
            idx : idx + SEQ_LEN
        ]

        u_seq = self.controls[
            idx : idx + SEQ_LEN
        ]

        return (

            torch.tensor(
                x0,
                dtype=torch.float32
            ),

            torch.tensor(
                target,
                dtype=torch.float32
            ),

            torch.tensor(
                u_seq,
                dtype=torch.float32
            )
        )

# ============================================================
# DATASETS
# ============================================================

train_dataset = TimeSeriesDataset(
    train_states,
    train_u
)

val_dataset = TimeSeriesDataset(
    val_states,
    val_u
)

test_dataset = TimeSeriesDataset(
    test_states,
    test_u
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# ============================================================
# ODE FUNCTION
# ============================================================

class ODEFunc(nn.Module):

    def __init__(self):

        super().__init__()

        self.current_u = None

        self.net = nn.Sequential(

            nn.Linear(3, 64),

            nn.Tanh(),

            nn.Linear(64, 64),

            nn.Tanh(),

            nn.Linear(64, 2)
        )

    def forward(self, t, y):

        u = self.current_u

        y_u = torch.cat(
            [y, u],
            dim=1
        )

        return self.net(y_u)

# ============================================================
# NEURAL ODE MODEL
# ============================================================

class NeuralODE(nn.Module):

    def __init__(self):

        super().__init__()

        self.func = ODEFunc()

    def forward(self, y0, t):

        pred_y = odeint(

            self.func,

            y0,

            t,

            method='rk4'
        )

        return pred_y

# ============================================================
# MODEL
# ============================================================

model = NeuralODE().to(device)

# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode='min',

    factor=0.5,

    patience=10
)

# ============================================================
# TIME VECTOR
# ============================================================

dt = 0.02

print("dt =", dt)

t_span = torch.linspace(

    0,

    dt * (SEQ_LEN - 1),

    SEQ_LEN

).to(device)

# ============================================================
# TRAINING
# ============================================================

EPOCHS = 200

best_val = np.inf

train_losses = []
val_losses = []

start_time = time.time()

for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    total_train_loss = 0

    for x0, target, u_seq in train_loader:

        x0 = x0.to(device)

        target = target.to(device)

        model.func.current_u = (

            u_seq[:, 0]
            .unsqueeze(1)
            .to(device)
        )

        optimizer.zero_grad()

        pred = model(
            x0,
            t_span
        )

        # [SEQ, BATCH, 2]
        # -> [BATCH, SEQ, 2]

        pred = pred.permute(1, 0, 2)

        loss = criterion(
            pred,
            target
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = (
        total_train_loss / len(train_loader)
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    total_val_loss = 0

    with torch.no_grad():

        for x0, target, u_seq in val_loader:

            x0 = x0.to(device)

            target = target.to(device)

            model.func.current_u = (

                u_seq[:, 0]
                .unsqueeze(1)
                .to(device)
            )

            pred = model(
                x0,
                t_span
            )

            pred = pred.permute(1, 0, 2)

            loss = criterion(
                pred,
                target
            )

            total_val_loss += loss.item()

    avg_val_loss = (
        total_val_loss / len(val_loader)
    )

    scheduler.step(avg_val_loss)

    train_losses.append(avg_train_loss)

    val_losses.append(avg_val_loss)

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if avg_val_loss < best_val:

        best_val = avg_val_loss

# ========================================================
# SAVE BEST MODEL AS ONNX
# ========================================================

    if avg_val_loss < best_val:

        best_val = avg_val_loss

        model.eval()

        dummy_y0 = torch.randn(1, 2).to(device)

        model.func.current_u = (
            torch.randn(1, 1).to(device)
        )

        torch.onnx.export(

            model,

            (dummy_y0, t_span),

            "best_model.onnx",

            export_params=True,

            opset_version=11,

            do_constant_folding=True,

            input_names=[
                'initial_state',
                'time_vector'
            ],

            output_names=[
                'predicted_states'
            ],

            dynamic_axes={

                'initial_state': {
                    0: 'batch_size'
                },

                'predicted_states': {
                    1: 'batch_size'
                }
            }
        )

        print("Best ONNX model saved!")
    # ========================================================
    # PRINT EVERY 10 EPOCHS
    # ========================================================

    if (epoch + 1) % 10 == 0:

        elapsed = time.time() - start_time

        print(

            f"Epoch {epoch+1}/{EPOCHS} | "

            f"Train Loss: {avg_train_loss:.6f} | "

            f"Val Loss: {avg_val_loss:.6f} | "

            f"Time: {elapsed:.2f} sec"
        )

    if (avg_train_loss < 0.1):
        print("Breaking early, train loss is less than 0.1%")
        break

# ============================================================
# TOTAL TRAINING TIME
# ============================================================

total_time = time.time() - start_time

minutes = total_time / 60

print(
    f"\nTotal Training Time: "
    f"{minutes:.2f} minutes"
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

best_model = NeuralODE().to(device)

# ============================================================
# TEST EVALUATION
# ============================================================

best_model.eval()

predictions = []
targets = []

with torch.no_grad():

    for x0, target, u_seq in test_loader:

        x0 = x0.to(device)

        model.func.current_u = (

            u_seq[:, 0]
            .unsqueeze(1)
            .to(device)
        )

        pred = model(
            x0,
            t_span
        )

        pred = pred.permute(1, 0, 2)

        predictions.append(
            pred.cpu().numpy()
        )

        targets.append(
            target.numpy()
        )

predictions = np.concatenate(
    predictions,
    axis=0
)

targets = np.concatenate(
    targets,
    axis=0
)

# ============================================================
# INVERSE TRANSFORM
# ============================================================

pred_flat = predictions.reshape(-1, 2)

target_flat = targets.reshape(-1, 2)

pred_inv = state_scaler.inverse_transform(
    pred_flat
)

target_inv = state_scaler.inverse_transform(
    target_flat
)

# ============================================================
# Error Metrics
# ============================================================

mse_phi = np.mean(
    (pred_inv[:, 0] - target_inv[:, 0]) ** 2
)
print(f"Test MSE of Phi: {mse_phi:.6f}")

mse_phid = np.mean(
    (pred_inv[:, 1] - target_inv[:, 1]) ** 2
)
print(f"Test MSE of Phid: {mse_phid:.6f}")

# ============================================================
# PLOTS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# EXTRACT CLEAN 1D SIGNALS
# ============================================================

true_phi  = target_inv[:, 0].reshape(-1)
pred_phi  = pred_inv[:, 0].reshape(-1)

true_phid = target_inv[:, 1].reshape(-1)
pred_phid = pred_inv[:, 1].reshape(-1)

# Sequential x-axis
x_phi  = np.arange(len(true_phi))
x_phid = np.arange(len(true_phid))

# ============================================================
# FIGURE
# ============================================================

plt.figure(figsize=(12, 5))

# ============================================================
# PHI
# ============================================================

plt.subplot(1, 2, 1)

plt.plot(
    x_phi,
    true_phi,
    label='True Phi',
    linewidth=1
)

plt.plot(
    x_phi,
    pred_phi,
    '--',
    label='Pred Phi',
    linewidth=1
)

plt.xlabel("Samples")
plt.ylabel("Angular Position")
plt.title("Phi Prediction")

plt.legend()
plt.grid(True)

# ============================================================
# PHID
# ============================================================

plt.subplot(1, 2, 2)

plt.plot(
    x_phid,
    true_phid,
    label='True Phid',
    linewidth=1
)

plt.plot(
    x_phid,
    pred_phid,
    '--',
    label='Pred Phid',
    linewidth=1
)

plt.xlabel("Samples")
plt.ylabel("Angular Velocity")
plt.title("Phid Prediction")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# ============================================================
# Error Plots
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    (pred_inv - target_inv).reshape(-1, 2)[:, 0],
    '--',
    label='Error Phi'
)

plt.xlabel("Samples")

plt.ylabel("Error in Angular Position")

plt.legend()

plt.grid()

plt.tight_layout()

plt.subplot(1, 2, 2)

plt.plot(
    (pred_inv - target_inv).reshape(-1, 2)[:, 1],
    '--',
    label='Error Phid'
)

plt.xlabel("Samples")

plt.ylabel("Error in Angular Velocity")

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()