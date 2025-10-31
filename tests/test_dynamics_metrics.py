import torch
from src.evaluation.dynamics_metrics import (
    velocity_mse,
    acceleration_mse,
    kinetic_energy_mse,
    temporal_frame_difference_error,
)


def test_velocity_acceleration_ke_simple():
    # Create a simple scenario: constant velocity ground-truth, simulated has small offset
    T = 5
    N = 4
    D = 3
    # GT: particle positions moving at velocity 0.1 along x
    gt = torch.zeros(T, N, D)
    for t in range(T):
        gt[t, :, 0] = 0.1 * t

    # Sim: slightly faster
    sim = torch.zeros_like(gt)
    for t in range(T):
        sim[t, :, 0] = 0.12 * t

    v_mse = velocity_mse(sim, gt, dt=1.0)
    # velocity difference is 0.02 -> squared = 4e-4 per-dim, per-particle, per-time
    assert v_mse > 0

    a_mse = acceleration_mse(sim, gt, dt=1.0)
    # constant velocities -> zero accelerations for GT, sim -> acceleration MSE small
    assert a_mse >= 0

    ke_mse = kinetic_energy_mse(sim, gt, mass=None, dt=1.0)
    assert ke_mse >= 0


def test_temporal_frame_diff_error():
    # Small synthetic video: one bright pixel moves to the right each frame
    T = 4
    C, H, W = 1, 8, 8
    gt = torch.zeros(T, C, H, W)
    sim = torch.zeros_like(gt)
    for t in range(T):
        x = min(W - 1, t)
        gt[t, 0, 4, x] = 1.0
        sim[t, 0, 4, min(W - 1, x + 1)] = 1.0  # sim is shifted by 1 px

    err = temporal_frame_difference_error(sim, gt)
    assert err > 0
