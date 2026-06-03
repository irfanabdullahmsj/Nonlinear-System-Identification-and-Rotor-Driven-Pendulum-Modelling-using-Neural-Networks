"""
model.py
========
Residual neural network for the hybrid physics-NN model.
"""

import torch
import torch.nn as nn


class ImprovedHybridNN(nn.Module):
    """
    Fully-connected network that predicts the residual
    (Δphi, Δphi_dot) not captured by the physics model.

    Architecture
    ------------
    Linear(input_size → 64) → SiLU
    Linear(64 → 64)          → SiLU
    Linear(64 → 32)          → SiLU
    Linear(32 → 2)

    Parameters
    ----------
    input_size : number of input features
                 (= len(delays) * 3)
    """

    def __init__(self, input_size: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.SiLU(),

            nn.Linear(64, 64),
            nn.SiLU(),

            nn.Linear(64, 32),
            nn.SiLU(),

            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# QUICK SANITY CHECK
# ============================================================

if __name__ == "__main__":
    from config import input_size
    net = ImprovedHybridNN(input_size)
    print(net)
    dummy = torch.randn(4, input_size)
    out   = net(dummy)
    print("Output shape:", out.shape)   # should be (4, 2)