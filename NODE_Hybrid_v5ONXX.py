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

# ============================================================
# EXTRACT SIGNALS
# ============================================================

train_phi  = train_data['PA'].squeeze()
train_phid = train_data['PAD'].squeeze()
train_u    = train_data['u'].squeeze()
train_x    = train_data['x'].squeeze()

val_phi  = val_data['PA'].squeeze()
val_phid = val_data['PAD'].squeeze()
val_u    = val_data['u'].squeeze()
val_x    = val_data['x'].squeeze()

test_phi  = test_data['PA'].squeeze()
test_phid = test_data['PAD'].squeeze()
test_u    = test_data['u'].squeeze()
test_x    = test_data['x'].squeeze()

# ============================================================
# STATES
# ============================================================

train_states = np.column_stack(
    (train_phi, train_phid)
)

val_states = np.column_stack(
    (val_phi, val_phid)
)

test_states = np.column_stack(
    (test_phi, test_phid)
)

# ============================================================
# NORMALIZATION
# ============================================================

state_scaler = StandardScaler()

train_states_scaled = state_scaler.fit_transform(
    train_states
)

val_states_scaled = state_scaler.transform(
    val_states
)

test_states_scaled = state_scaler.transform(
    test_states
)

# ------------------------------------------------------------

u_scaler = StandardScaler()

train_u_scaled = u_scaler.fit_transform(
    train_u.reshape(-1, 1)
).flatten()

val_u_scaled = u_scaler.transform(
    val_u.reshape(-1, 1)
).flatten()

test_u_scaled = u_scaler.transform(
    test_u.reshape(-1, 1)
).flatten()

# ------------------------------------------------------------

x_scaler = StandardScaler()

train_x_scaled = x_scaler.fit_transform(
    train_x.reshape(-1, 1)
).flatten()

val_x_scaled = x_scaler.transform(
    val_x.reshape(-1, 1)
).flatten()

test_x_scaled = x_scaler.transform(
    test_x.reshape(-1, 1)
).flatten()

# ============================================================
# TRAIN / VAL / TEST
# ============================================================

train_states = train_states_scaled
val_states   = val_states_scaled
test_states  = test_states_scaled

train_u = train_u_scaled
val_u   = val_u_scaled
test_u  = test_u_scaled

train_x = train_x_scaled
val_x   = val_x_scaled
test_x  = test_x_scaled

# ============================================================
# DATASET
# ============================================================

SEQ_LEN = 20

class TimeSeriesDataset(Dataset):

    def __init__(
        self,
        states,
        controls,
        inputs
    ):

        self.states = states
        self.controls = controls
        self.inputs = inputs

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

        x_seq = self.inputs[
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
            ),

            torch.tensor(
                x_seq,
                dtype=torch.float32
            )
        )

# ============================================================
# DATASETS
# ============================================================

train_dataset = TimeSeriesDataset(
    train_states,
    train_u,
    train_x
)

val_dataset = TimeSeriesDataset(
    val_states,
    val_u,
    val_x
)

test_dataset = TimeSeriesDataset(
    test_states,
    test_u,
    test_x
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

        self.u_sequence = None
        self.x_sequence = None

        self.dt = 0.02

        self.net = nn.Sequential(

            nn.Linear(4, 64),

            nn.Tanh(),

            nn.Linear(64, 64),

            nn.Tanh(),

            nn.Linear(64, 2)
        )

    def forward(self, t, y):

        # Time index
        idx = int(torch.clamp(
            torch.round(t / self.dt),
            0,
            SEQ_LEN - 1
        ).item())

        # Get time-varying inputs
        u = self.u_sequence[:, idx].unsqueeze(1)

        x = self.x_sequence[:, idx].unsqueeze(1)

        # Concatenate
        y_u_x = torch.cat(
            [y, u, x],
            dim=1
        )

        return self.net(y_u_x)

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

    for x0, target, u_seq, x_seq in train_loader:

        x0 = x0.to(device)

        target = target.to(device)

        # Assign sequences
        model.func.u_sequence = (
            u_seq.to(device)
        )

        model.func.x_sequence = (
            x_seq.to(device)
        )

        optimizer.zero_grad()

        pred = model(
            x0,
            t_span
        )

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

        for x0, target, u_seq, x_seq in val_loader:

            x0 = x0.to(device)

            target = target.to(device)

            model.func.u_sequence = (
                u_seq.to(device)
            )

            model.func.x_sequence = (
                x_seq.to(device)
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

        # Save PyTorch weights
        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        model.eval()

        # Dummy input
        dummy_y0 = torch.randn(
            1,
            2
        ).to(device)

        dummy_u_seq = torch.randn(
            1,
            SEQ_LEN
        ).to(device)

        dummy_x_seq = torch.randn(
            1,
            SEQ_LEN
        ).to(device)

        # Assign dummy sequences
        model.func.u_sequence = dummy_u_seq

        model.func.x_sequence = dummy_x_seq

        # Export ONNX
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

        print(
            f"Best model saved! "
            f"Validation Loss: "
            f"{best_val:.6f}"
        )

    # ========================================================
    # PRINT
    # ========================================================

    if (epoch + 1) % 10 == 0:

        elapsed = time.time() - start_time

        print(

            f"Epoch {epoch+1}/{EPOCHS} | "

            f"Train Loss: "
            f"{avg_train_loss:.6f} | "

            f"Val Loss: "
            f"{avg_val_loss:.6f} | "

            f"Time: "
            f"{elapsed:.2f} sec"
        )

    # ========================================================
    # EARLY STOP
    # ========================================================

    if avg_train_loss < 0.1:

        print(
            "Breaking early, "
            "train loss < 0.1"
        )

        break

# ============================================================
# TRAINING TIME
# ============================================================

total_time = time.time() - start_time

print(
    f"\nTotal Training Time: "
    f"{total_time/60:.2f} minutes"
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

best_model = NeuralODE().to(device)

best_model.load_state_dict(

    torch.load(
        "best_model.pth",
        map_location=device
    )
)

best_model.eval()

# ============================================================
# TESTING
# ============================================================

predictions = []
targets = []

with torch.no_grad():

    for x0, target, u_seq, x_seq in test_loader:

        x0 = x0.to(device)

        best_model.func.u_sequence = (
            u_seq.to(device)
        )

        best_model.func.x_sequence = (
            x_seq.to(device)
        )

        pred = best_model(
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
# ERROR METRICS
# ============================================================

mse_phi = np.mean(
    (pred_inv[:, 0] - target_inv[:, 0]) ** 2
)

mse_phid = np.mean(
    (pred_inv[:, 1] - target_inv[:, 1]) ** 2
)

print(f"Test MSE Phi : {mse_phi:.6f}")

print(f"Test MSE Phid: {mse_phid:.6f}")

# ============================================================
# PLOTS
# ============================================================

true_phi = target_inv[:, 0].reshape(-1)

pred_phi = pred_inv[:, 0].reshape(-1)

true_phid = target_inv[:, 1].reshape(-1)

pred_phid = pred_inv[:, 1].reshape(-1)

# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

# ============================================================
# PHI
# ============================================================

plt.subplot(1, 2, 1)

plt.plot(
    true_phi,
    label='True Phi'
)

plt.plot(
    pred_phi,
    '--',
    label='Pred Phi'
)

plt.title("Phi Prediction")

plt.xlabel("Samples")

plt.ylabel("Angular Position")

plt.grid(True)

plt.legend()

# ============================================================
# PHID
# ============================================================

plt.subplot(1, 2, 2)

plt.plot(
    true_phid,
    label='True Phid'
)

plt.plot(
    pred_phid,
    '--',
    label='Pred Phid'
)

plt.title("Phid Prediction")

plt.xlabel("Samples")

plt.ylabel("Angular Velocity")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()

# ============================================================
# ERROR PLOTS
# ============================================================

plt.figure(figsize=(12, 5))

# ------------------------------------------------------------

plt.subplot(1, 2, 1)

plt.plot(
    pred_inv[:, 0] - target_inv[:, 0],
    '--',
    label='Phi Error'
)

plt.xlabel("Samples")

plt.ylabel("Error")

plt.grid(True)

plt.legend()

# ------------------------------------------------------------

plt.subplot(1, 2, 2)

plt.plot(
    pred_inv[:, 1] - target_inv[:, 1],
    '--',
    label='Phid Error'
)

plt.xlabel("Samples")

plt.ylabel("Error")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()