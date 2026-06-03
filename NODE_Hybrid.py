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

data = loadmat(
    r"I:\h_da\Summer 2026\Team_Project\Data_Preprocessing\Reduced_Exp_12May_1457_upsamp_filt.mat"
)

phi  = data['PA'].squeeze()
phid = data['PAD'].squeeze()
u    = data['u'].squeeze()
t    = data['t'].squeeze()

# ============================================================
# STATES
# ============================================================

states = np.column_stack((phi, phid))

# ============================================================
# NORMALIZATION
# ============================================================

state_scaler = StandardScaler()

states_scaled = state_scaler.fit_transform(
    states
)

u_scaler = StandardScaler()

u_scaled = u_scaler.fit_transform(
    u.reshape(-1, 1)
).flatten()

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

N = len(states_scaled)

train_end = int(0.70 * N)
val_end   = int(0.85 * N)

train_states = states_scaled[:train_end]
val_states   = states_scaled[train_end:val_end]
test_states  = states_scaled[val_end:]

train_u = u_scaled[:train_end]
val_u   = u_scaled[train_end:val_end]
test_u  = u_scaled[val_end:]

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
    shuffle=True
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

dt = t[1] - t[0]

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

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

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

model.load_state_dict(
    torch.load("best_model.pth")
)

# ============================================================
# TEST EVALUATION
# ============================================================

model.eval()

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
# PLOTS
# ============================================================

plt.figure(figsize=(12, 5))

# ============================================================
# PHI
# ============================================================

plt.subplot(1, 2, 1)

plt.plot(
    target_inv[:, 0],
    label='True Phi'
)

plt.plot(
    pred_inv[:, 0],
    '--',
    label='Pred Phi'
)

plt.xlabel("Samples")

plt.ylabel("Angular Position")

plt.legend()

plt.grid()

# ============================================================
# PHID
# ============================================================

plt.subplot(1, 2, 2)

plt.plot(
    target_inv[:, 1],
    label='True Phid'
)

plt.plot(
    pred_inv[:, 1],
    '--',
    label='Pred Phid'
)

plt.xlabel("Samples")

plt.ylabel("Angular Velocity")

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()