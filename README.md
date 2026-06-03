# Neural System Identification & MPC Control — Rotor-Actuated Pendulum

Hybrid physics-informed ML for nonlinear system identification and Model Predictive Control of a rotor-actuated pendulum.

---

## What It Does

- Identifies pendulum dynamics using two approaches: **Neural ODE** and **Residual Gray-Box** (physics + NN correction)
- Predicts angular position (φ) and velocity (φ̇) from motor input u and slider position x
- Controls the pendulum using **RK4-based MPC** with both pure physics and hybrid prediction models

---

## Approaches

| | Neural ODE | Residual Gray-Box |
|---|---|---|
| **Idea** | NN learns ẋ = f(x,u), integrated via RK4 | Physics model + NN corrects residual error |
| **RMSE φ** | ~1.05° | **~0.45°** |
| **Recursive stability** | Moderate | Better |
| **Winner** | ❌ | ✅ |

> The residual model wins — strong physics knowledge of this plant leaves little room for a purely data-driven approach.

---

## MPC Results

| Controller | Scenario | Settling Time |
|---|---|---|
| RK4 Physics MPC | 20° → 0° | ~2s |
| Hybrid MPC | 20° → 0° | **~0.1s** |
| RK4 Physics MPC | 0° → 5° tracking | Clean |
| Hybrid MPC | 0° → 5° tracking | Steady-state offset |

---

## Stack

`Python` · `PyTorch` · `torchdiffeq` · `MATLAB` · `NumPy/SciPy` · `ONNX` · `Matplotlib`

---

## Team

| Member | Role |
|---|---|
| Irfan | Data preprocessing, Neural ODE, MPC pipeline |
| Benedikt | Neural ODE, architecture optimization, state-space modelling |
| Lalith | Residual gray-box model, recursive simulation, stability |
| Enrique | Data acquisition, physical model identification |

---

## Next Steps

- [ ] Deploy Hybrid MPC on physical plant (Hardware-in-Loop)
- [ ] Explore model-based reinforcement learning for control
- [ ] Improve recursive stability beyond 30s horizon

---

## References

1. Luo et al., *MPC of Nonlinear Processes Using Neural ODE Models*, Comp. Chem. Eng., 2023
2. Matzakos & Sfyrakis, *Comparing PINN and Neural ODE Approaches*, arXiv:2603.26921, 2026
