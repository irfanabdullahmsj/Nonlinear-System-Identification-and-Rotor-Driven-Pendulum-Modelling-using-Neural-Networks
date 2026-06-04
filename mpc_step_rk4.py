import numpy as np
import matplotlib.pyplot as plt

from pendulum_model import rk4_step, Ts

# ============================================================
# MPC SETTINGS
# ============================================================

Np = 50

Q_phi = 5000.0
Q_pad = 50.0
R_u = 0.0001

# Positive inputs only
u_min = 0.0
u_max = 40.0

candidate_inputs = np.linspace(
    u_min,
    u_max,
    81
)

# ============================================================
# MPC CONTROLLER
# ============================================================

def mpc_control(
    phi,
    phi_dot,
    phi_ref
):

    best_cost = np.inf
    best_u = 0.0

    for u in candidate_inputs:

        phi_pred = phi
        phi_dot_pred = phi_dot

        cost = 0.0

        for _ in range(Np):

            phi_pred, phi_dot_pred = rk4_step(
                phi_pred,
                phi_dot_pred,
                u,
                Ts
            )

            cost += (
                Q_phi * (phi_pred - phi_ref)**2
                + Q_pad * phi_dot_pred**2
                + R_u * u**2
            )

        if cost < best_cost:

            best_cost = cost
            best_u = u

    return best_u


# ============================================================
# SIMULATION SETTINGS
# ============================================================

Tsim = 5.0
Nsim = int(Tsim / Ts)

phi_hist = np.zeros(Nsim)
phi_dot_hist = np.zeros(Nsim)
u_hist = np.zeros(Nsim)

# ============================================================
# REFERENCE
# ============================================================

phi_ref_deg = 90.0
phi_ref = np.deg2rad(phi_ref_deg)

# ============================================================
# INITIAL CONDITION
# ============================================================

phi_hist[0] = np.deg2rad(0.0)
phi_dot_hist[0] = 0.0

# ============================================================
# SIMULATION LOOP
# ============================================================

for k in range(Nsim - 1):

    phi = phi_hist[k]
    phi_dot = phi_dot_hist[k]

    u = mpc_control(
        phi,
        phi_dot,
        phi_ref
    )

    u_hist[k] = u

    if k % 100 == 0:

        print(
            f"Step {k:5d} | "
            f"phi = {np.rad2deg(phi):8.3f} deg | "
            f"PAD = {np.rad2deg(phi_dot):8.3f} deg/s | "
            f"u = {u:8.3f}"
        )

    phi_next, phi_dot_next = rk4_step(
        phi,
        phi_dot,
        u,
        Ts
    )

    phi_hist[k + 1] = phi_next
    phi_dot_hist[k + 1] = phi_dot_next


# ============================================================
# PLOTS
# ============================================================

t = np.arange(Nsim) * Ts

plt.figure(figsize=(12, 8))

# ------------------------------------------------------------
# PA
# ------------------------------------------------------------

plt.subplot(3, 1, 1)

plt.plot(
    t,
    np.rad2deg(phi_hist),
    label="PA"
)

plt.axhline(
    phi_ref_deg,
    color="r",
    linestyle="--",
    label="Reference"
)

plt.ylabel("PA [deg]")
plt.title(f"RK4 MPC Tracking ({phi_ref_deg} deg)")
plt.grid(True)
plt.legend()

# ------------------------------------------------------------
# PAD
# ------------------------------------------------------------

plt.subplot(3, 1, 2)

plt.plot(
    t,
    np.rad2deg(phi_dot_hist)
)

plt.ylabel("PAD [deg/s]")
plt.grid(True)

# ------------------------------------------------------------
# CONTROL INPUT
# ------------------------------------------------------------

plt.subplot(3, 1, 3)

plt.plot(
    t,
    u_hist
)

plt.ylabel("u")
plt.xlabel("Time [s]")
plt.grid(True)

plt.tight_layout()
plt.show()
