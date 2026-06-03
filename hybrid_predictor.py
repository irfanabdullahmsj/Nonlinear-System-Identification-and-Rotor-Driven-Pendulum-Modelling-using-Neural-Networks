"""
hybrid_predictor.py
===================

Hybrid Physics + Neural Network predictor.

Usage:
-------
predictor = HybridPredictor(
    "improved_hybrid_model.pth",
    "x_scaler.pkl",
    "y_scaler.pkl"
)

phi_next, phi_dot_next = predictor.predict_next(
    phi_hist,
    phi_dot_hist,
    u_hist
)
"""

import sys
import os

project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)


import numpy as np
import torch
import joblib

from config import delays, Ts, device
from physics import rk4_step
from model import ImprovedHybridNN


class HybridPredictor:

    def __init__(
        self,
        model_path,
        x_scaler_path,
        y_scaler_path
    ):

        checkpoint = torch.load(
            model_path,
            map_location=device
        )

        self.model = ImprovedHybridNN(
            checkpoint["input_size"]
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.x_scaler = joblib.load(
            x_scaler_path
        )

        self.y_scaler = joblib.load(
            y_scaler_path
        )

    def predict_next(
        self,
        phi_hist,
        phi_dot_hist,
        u_hist
    ):
        """
        Parameters
        ----------
        phi_hist     : list/array containing delayed phi values
        phi_dot_hist : list/array containing delayed phi_dot values
        u_hist       : list/array containing delayed control inputs

        Expected ordering:
        index 0 -> delay 0
        index 1 -> delay 1
        index 2 -> delay 2
        index 3 -> delay 5
        index 4 -> delay 10

        Returns
        -------
        phi_next
        phi_dot_next
        """

        features = []

        for i in range(len(delays)):

            features.extend([
                phi_hist[i],
                phi_dot_hist[i],
                u_hist[i]
            ])

        X = np.array(
            features,
            dtype=np.float32
        ).reshape(1, -1)

        X_scaled = self.x_scaler.transform(X)

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        ).to(device)

        # current state
        phi_now = phi_hist[0]
        phi_dot_now = phi_dot_hist[0]
        u_now = u_hist[0]

        # physics prediction
        phi_phys, phi_dot_phys = rk4_step(
            phi_now,
            phi_dot_now,
            u_now,
            Ts
        )

        # neural residual
        with torch.no_grad():

            residual_scaled = (
                self.model(X_tensor)
                .cpu()
                .numpy()
            )

        residual = (
            self.y_scaler
            .inverse_transform(
                residual_scaled
            )[0]
        )

        # hybrid prediction
        phi_next = (
            phi_phys
            + residual[0]
        )

        phi_dot_next = (
            phi_dot_phys
            + residual[1]
        )

        return (
            phi_next,
            phi_dot_next
        )