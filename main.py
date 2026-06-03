"""
main.py
=======
Entry point — runs the full pipeline in order:
    1. Print configuration summary
    2. Train the hybrid model  (train.py)
    3. Evaluate and plot       (evaluate.py)

Usage
-----
    python main.py
"""

import torch
import config as cfg


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

print("=" * 50)
print("DEVICE CONFIGURATION")
print("=" * 50)
print("Using device :", cfg.device)

print("\nDelay configuration :", cfg.delays)
print("Maximum delay        :", cfg.max_delay)
print("Input size           :", cfg.input_size)

print("\nPhysical parameters:")
print(f"  a0 = {cfg.a0:.6f}")
print(f"  a1 = {cfg.a1:.6f}")
print(f"  a2 = {cfg.a2:.6f}")


# ============================================================
# TRAIN
# ============================================================

import train          # executes training on import


# ============================================================
# EVALUATE
# ============================================================

import evaluate       # executes evaluation and shows plots