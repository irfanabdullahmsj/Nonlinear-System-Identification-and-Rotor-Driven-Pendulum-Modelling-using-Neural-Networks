import numpy as np
import torch
import joblib

from config import Ts, delays, device
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
        phid_hist,
        u_hist
    ):

        features = []

        for d in delays:

            features.extend([
                phi_hist[d],
                phid_hist[d],
                u_hist[d]
            ])

        X = np.array(
            features,
            dtype=np.float32
        ).reshape(1, -1)

        Xs = self.x_scaler.transform(X)

        X_tensor = torch.tensor(
            Xs,
            dtype=torch.float32
        ).to(device)

        phi_now = phi_hist[0]
        phid_now = phid_hist[0]
        u_now = u_hist[0]

        phi_phys, phid_phys = rk4_step(
            phi_now,
            phid_now,
            u_now,
            Ts
        )

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

        phi_next = (
            phi_phys
            + residual[0]
        )

        phid_next = (
            phid_phys
            + residual[1]
        )

        return phi_next, phid_next