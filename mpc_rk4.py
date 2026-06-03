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

u_min = -10.0
u_max = 10.0

candidate_inputs = np.linspace(
    u_min,
    u_max,
    81
)

# ============================================================
# MPC CONTROLLER
# ============================================================

def mpc_control(phi, phi_dot):

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
                Q_phi * phi_pred**2
                + Q_pad * phi_dot_pred**2
                + R_u * u**2
            )

        if cost < best_cost:

            best_cost = cost
            best_u = u

    return best_u


# ============================================================
# CLOSED LOOP SIMULATION
# ============================================================

Tsim = 5.0

Nsim = int(Tsim / Ts)

phi_hist = np.zeros(Nsim)
phi_dot_hist = np.zeros(Nsim)
u_hist = np.zeros(Nsim)
for k in range(Nsim - 1):

    phi = phi_hist[k]
    phi_dot = phi_dot_hist[k]

    u = mpc_control(
        phi,
        phi_dot
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

# ------------------------------------------------------------
# INITIAL CONDITION
# ------------------------------------------------------------

phi_hist[0] = np.deg2rad(90)
phi_dot_hist[0] = 0.0
# ============================================================
# MOTOR CURVE CHECK
# ============================================================

u_test = np.linspace(-20, 20, 500)

torque_test = (
    -0.00000992 * u_test**3
    + 0.00105529 * u_test**2
    - 0.00489381 * u_test
)

plt.figure(figsize=(8,5))

plt.plot(
    u_test,
    torque_test,
    linewidth=2
)

plt.axhline(
    0,
    color='k',
    linestyle='--'
)

plt.grid(True)

plt.xlabel("u")

plt.ylabel("Motor Torque")

plt.title("Mr_cubic(u)")

#plt.show()
# ============================================================
# SIMULATION LOOP
# ============================================================

for k in range(Nsim - 1):

    phi = phi_hist[k]
    phi_dot = phi_dot_hist[k]

    u = mpc_control(
        phi,
        phi_dot
    )

    u_hist[k] = u

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

plt.subplot(3, 1, 1)

plt.plot(
    t,
    np.rad2deg(phi_hist)
)

plt.ylabel("PA [deg]")
plt.title("RK4 MPC")

plt.grid(True)

plt.subplot(3, 1, 2)

plt.plot(
    t,
    np.rad2deg(phi_dot_hist)
)

plt.ylabel("PAD [deg/s]")

plt.grid(True)

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