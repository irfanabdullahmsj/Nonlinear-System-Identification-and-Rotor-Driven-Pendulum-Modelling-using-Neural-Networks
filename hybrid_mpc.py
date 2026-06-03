import numpy as np
import matplotlib.pyplot as plt

from hybrid_predictor import HybridPredictor
from config import Ts, delays

predictor = HybridPredictor(
    "improved_hybrid_model.pth",
    "x_scaler.pkl",
    "y_scaler.pkl"
)

phi_next, phid_next = predictor.predict_next(
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
)

print("phi_next =", phi_next)
print("phi_dot_next =", phid_next)


# MPC settings below...
Np = 20

# ============================================================
# LOAD HYBRID MODEL
# ============================================================

predictor = HybridPredictor(
    "improved_hybrid_model.pth",
    "x_scaler.pkl",
    "y_scaler.pkl"
)

# ============================================================
# MPC SETTINGS
# ============================================================

Np = 20

Q_phi = 5000.0
Q_pad = 50.0
R_u = 0.001

u_min = -10.0
u_max = 10.0

candidate_inputs = np.linspace(
    u_min,
    u_max,
    81
)

phi_ref = np.deg2rad(5.0)

# ============================================================
# MPC CONTROLLER
# ============================================================

def mpc_control(
    phi_hist,
    phi_dot_hist,
    u_hist
):

    best_cost = np.inf
    best_u = 0.0

    for u_candidate in candidate_inputs:

        phi_buffer = list(phi_hist)
        phi_dot_buffer = list(phi_dot_hist)
        u_buffer = list(u_hist)

        cost = 0.0

        for _ in range(Np):

            # current input assumed constant
            u_buffer[0] = u_candidate

            phi_pred, phi_dot_pred = predictor.predict_next(
                phi_buffer,
                phi_dot_buffer,
                u_buffer
            )

            cost += (
                Q_phi * (phi_pred - phi_ref) ** 2
                + Q_pad * phi_dot_pred ** 2
                + R_u * u_candidate ** 2
            )

            # update delay buffers
            phi_buffer = [
                phi_pred,
                phi_buffer[0],
                phi_buffer[1],
                phi_buffer[2],
                phi_buffer[3]
            ]

            phi_dot_buffer = [
                phi_dot_pred,
                phi_dot_buffer[0],
                phi_dot_buffer[1],
                phi_dot_buffer[2],
                phi_dot_buffer[3]
            ]

            u_buffer = [
                u_candidate,
                u_buffer[0],
                u_buffer[1],
                u_buffer[2],
                u_buffer[3]
            ]

        if cost < best_cost:

            best_cost = cost
            best_u = u_candidate

    return best_u


# ============================================================
# SIMULATION SETTINGS
# ============================================================

Tsim = 2.0
Nsim = int(Tsim / Ts)

phi_hist_sim = np.zeros(Nsim)
phi_dot_hist_sim = np.zeros(Nsim)
u_sim = np.zeros(Nsim)

# Initial condition
phi_hist_sim[0] = np.deg2rad(0.0)
phi_dot_hist_sim[0] = 0.0

# ============================================================
# INITIAL DELAY BUFFERS
# ============================================================

phi_buffer = [phi_hist_sim[0]] * len(delays)
phi_dot_buffer = [0.0] * len(delays)
u_buffer = [0.0] * len(delays)

for u in [-10,-5,0,5,10]:

    phi_test, pad_test = predictor.predict_next(
        phi_buffer,
        phi_dot_buffer,
        [u,u,u,u,u]
    )

    print(u, phi_test, pad_test)
# ============================================================
# CLOSED LOOP SIMULATION
# ============================================================

for k in range(Nsim - 1):

    u = mpc_control(
        phi_buffer,
        phi_dot_buffer,
        u_buffer
    )

    u_sim[k] = u

    u_buffer[0] = u

    phi_next, phi_dot_next = predictor.predict_next(
        phi_buffer,
        phi_dot_buffer,
        u_buffer
    )

    phi_hist_sim[k + 1] = phi_next
    phi_dot_hist_sim[k + 1] = phi_dot_next

    phi_buffer = [
        phi_next,
        phi_buffer[0],
        phi_buffer[1],
        phi_buffer[2],
        phi_buffer[3]
    ]

    phi_dot_buffer = [
        phi_dot_next,
        phi_dot_buffer[0],
        phi_dot_buffer[1],
        phi_dot_buffer[2],
        phi_dot_buffer[3]
    ]

    u_buffer = [
        u,
        u_buffer[0],
        u_buffer[1],
        u_buffer[2],
        u_buffer[3]
    ]

    if k % 100 == 0:

        print(
            f"Step {k:5d} | "
            f"PA = {np.rad2deg(phi_next):8.3f} deg | "
            f"PAD = {np.rad2deg(phi_dot_next):8.3f} deg/s | "
            f"u = {u:8.3f}"
        )

# ============================================================
# PLOTS
# ============================================================

t = np.arange(Nsim) * Ts

plt.figure(figsize=(12, 8))

plt.subplot(3,1,1)
plt.plot(t, np.rad2deg(phi_hist_sim))
plt.ylabel("PA [deg]")
plt.title("Hybrid MPC Stabilization")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t, np.rad2deg(phi_dot_hist_sim))
plt.ylabel("PAD [deg/s]")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(t, u_sim)
plt.ylabel("u")
plt.xlabel("Time [s]")
plt.grid(True)

plt.tight_layout()
plt.subplot(3,1,1)
plt.plot(t, np.rad2deg(phi_hist_sim), label="PA")
plt.axhline(5, color='r', linestyle='--', label='Reference')
plt.ylabel("PA [deg]")
plt.title("Hybrid MPC Tracking")
plt.legend()
plt.grid(True)
plt.show()