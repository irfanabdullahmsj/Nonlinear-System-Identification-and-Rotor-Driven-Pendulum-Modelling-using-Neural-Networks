import numpy as np

# ============================================================
# SAMPLE TIME
# ============================================================

Ts = 0.002

# ============================================================
# PHYSICAL PARAMETERS
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
print("\n==============================")
print("PENDULUM PARAMETERS")
print("==============================")

print(f"a0 = {a0:.6f}")
print(f"a1 = {a1:.6f}")
print(f"a2 = {a2:.6f}")

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
# DYNAMICS
# ============================================================

def dynamics(phi, phi_dot, u):

    phi_ddot = (
        -a1 * phi_dot
        -a0 * np.sin(phi)
        + Mr_cubic(u)
    ) / a2

    return phi_dot, phi_ddot


# ============================================================
# RK4 STEP
# ============================================================

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
print("\n==============================")
print("TORQUE COMPARISON")
print("==============================")

gravity_torque_20deg = (
    a0 * np.sin(
        np.deg2rad(20)
    )
)

motor_max = max(
    abs(
        -0.00000992 * (-10)**3
        + 0.00105529 * (-10)**2
        - 0.00489381 * (-10)
    ),
    abs(
        -0.00000992 * (10)**3
        + 0.00105529 * (10)**2
        - 0.00489381 * (10)
    )
)

print(
    f"Gravity term at 20 deg : {gravity_torque_20deg:.6f}"
)

print(
    f"Maximum motor torque   : {motor_max:.6f}"
)

print(
    f"Ratio gravity/motor    : "
    f"{gravity_torque_20deg/motor_max:.2f}"
)