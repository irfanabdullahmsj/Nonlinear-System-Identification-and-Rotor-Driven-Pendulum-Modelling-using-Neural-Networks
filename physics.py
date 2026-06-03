"""
physics.py
==========
Motor torque model and RK4 numerical integrator
for the pendulum dynamics.
"""

import numpy as np
from config import a0, a1, a2, Ts


# ============================================================
# MOTOR TORQUE MODEL  (cubic polynomial fit)
# ============================================================

def Mr_cubic(u: float) -> float:
    """Return motor torque [N·m] for PWM input u."""
    return (
        -0.00000992 * u**3
        + 0.00105529 * u**2
        - 0.00489381 * u
    )


# ============================================================
# CONTINUOUS DYNAMICS
# ============================================================

def dynamics(phi: float, phi_dot: float, u: float):
    """
    Equations of motion:
        phi_ddot = (-a1*phi_dot - a0*sin(phi) + Mr(u)) / a2

    Returns
    -------
    (d_phi, d_phi_dot) : derivatives
    """
    phi_ddot = (
        -a1 * phi_dot
        - a0 * np.sin(phi)
        + Mr_cubic(u)
    ) / a2
    return phi_dot, phi_ddot


# ============================================================
# RK4 INTEGRATOR
# ============================================================

def rk4_step(
    phi: float,
    phi_dot: float,
    u: float,
    Ts: float = Ts
):
    """
    One RK4 step forward.

    Parameters
    ----------
    phi     : current angle [rad]
    phi_dot : current angular velocity [rad/s]
    u       : motor input
    Ts      : sampling period [s]

    Returns
    -------
    (phi_next, phi_dot_next)
    """
    k1_phi, k1_dot = dynamics(phi, phi_dot, u)

    k2_phi, k2_dot = dynamics(
        phi     + 0.5 * Ts * k1_phi,
        phi_dot + 0.5 * Ts * k1_dot,
        u
    )

    k3_phi, k3_dot = dynamics(
        phi     + 0.5 * Ts * k2_phi,
        phi_dot + 0.5 * Ts * k2_dot,
        u
    )

    k4_phi, k4_dot = dynamics(
        phi     + Ts * k3_phi,
        phi_dot + Ts * k3_dot,
        u
    )

    coeff = Ts / 6.0

    phi_next     = phi     + coeff * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
    phi_dot_next = phi_dot + coeff * (k1_dot + 2*k2_dot + 2*k3_dot + k4_dot)

    return phi_next, phi_dot_next