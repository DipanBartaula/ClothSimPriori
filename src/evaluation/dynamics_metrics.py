import torch
import torch.nn.functional as F
from typing import Optional


def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        # (T, N, D) -> (1, T, N, D)
        return x.unsqueeze(0)
    return x


def velocity_mse(sim_pos: torch.Tensor, gt_pos: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """
    Mean-squared error between simulated and ground-truth velocities.

    Args:
        sim_pos, gt_pos: tensors of shape (T, N, D) or (B, T, N, D)
        dt: timestep between frames (default 1.0)

    Returns:
        Tensor of shape (B,) or scalar with per-batch mean velocity MSE.
    """
    sim = _ensure_batch(sim_pos)
    gt = _ensure_batch(gt_pos)
    assert sim.shape == gt.shape, "sim_pos and gt_pos must have same shape"
    # compute velocities (B, T-1, N, D)
    vs = (sim[:, 1:] - sim[:, :-1]) / dt
    vg = (gt[:, 1:] - gt[:, :-1]) / dt
    mse = F.mse_loss(vs, vg, reduction="none")
    # mean over time, particles, dims
    mse = mse.view(mse.shape[0], -1).mean(dim=1)
    return mse if mse.numel() > 1 else mse.squeeze(0)


def acceleration_mse(sim_pos: torch.Tensor, gt_pos: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """
    MSE between accelerations (second-order finite differences).
    """
    sim = _ensure_batch(sim_pos)
    gt = _ensure_batch(gt_pos)
    assert sim.shape == gt.shape, "sim_pos and gt_pos must have same shape"
    # velocities
    vs = (sim[:, 1:] - sim[:, :-1]) / dt
    vg = (gt[:, 1:] - gt[:, :-1]) / dt
    # accelerations (B, T-2, N, D)
    asim = (vs[:, 1:] - vs[:, :-1]) / dt
    agt = (vg[:, 1:] - vg[:, :-1]) / dt
    mse = F.mse_loss(asim, agt, reduction="none")
    mse = mse.view(mse.shape[0], -1).mean(dim=1)
    return mse if mse.numel() > 1 else mse.squeeze(0)


def kinetic_energy_mse(sim_pos: torch.Tensor, gt_pos: torch.Tensor, mass: Optional[torch.Tensor] = None, dt: float = 1.0) -> torch.Tensor:
    """
    MSE between kinetic energy time series of simulated and ground-truth particle sets.

    If mass is None, unit mass for each particle is assumed.
    Args:
        sim_pos, gt_pos: (T,N,D) or (B,T,N,D)
        mass: None or tensor (N,) or (B,N)
    Returns:
        per-batch MSE between KE time series (averaged over time and particles)
    """
    sim = _ensure_batch(sim_pos)
    gt = _ensure_batch(gt_pos)
    assert sim.shape == gt.shape, "sim_pos and gt_pos must have same shape"
    B, T, N, D = sim.shape
    vs = (sim[:, 1:] - sim[:, :-1]) / dt  # (B, T-1, N, D)
    vg = (gt[:, 1:] - gt[:, :-1]) / dt

    if mass is None:
        m = torch.ones((B, N), device=sim.device, dtype=sim.dtype)
    else:
        m = mass
        if m.dim() == 1:
            m = m.unsqueeze(0).expand(B, -1)

    # kinetic energy per particle per time: 0.5 * m * ||v||^2
    ke_sim = 0.5 * (m.unsqueeze(1) * (vs ** 2).sum(dim=-1))  # (B, T-1, N)
    ke_gt = 0.5 * (m.unsqueeze(1) * (vg ** 2).sum(dim=-1))

    mse = F.mse_loss(ke_sim, ke_gt, reduction="none")
    mse = mse.view(B, -1).mean(dim=1)
    return mse if mse.numel() > 1 else mse.squeeze(0)


def temporal_frame_difference_error(sim_video: torch.Tensor, gt_video: torch.Tensor) -> torch.Tensor:
    """
    Compare motion signals between two videos using frame differences.

    This metric computes the per-frame difference images (I_{t+1} - I_t)
    for both videos and returns the mean L2 difference between those
    motion maps. Inputs should be (B, T, C, H, W) or (T, C, H, W) or
    (C,H,W).
    """
    # Normalize shapes to (B, T, C, H, W)
    if sim_video.dim() == 4:
        sim_video = sim_video.unsqueeze(0)
        gt_video = gt_video.unsqueeze(0)
    elif sim_video.dim() == 3:
        # (T, C, H, W)
        sim_video = sim_video.unsqueeze(0)
        gt_video = gt_video.unsqueeze(0)

    assert sim_video.shape == gt_video.shape, "sim and gt videos must match shapes"
    B, T, C, H, W = sim_video.shape
    # frame diffs (B, T-1, C, H, W)
    diff_s = sim_video[:, 1:] - sim_video[:, :-1]
    diff_g = gt_video[:, 1:] - gt_video[:, :-1]

    mse = F.mse_loss(diff_s, diff_g, reduction="none")
    mse = mse.view(B, -1).mean(dim=1)
    return mse if mse.numel() > 1 else mse.squeeze(0)


__all__ = [
    "velocity_mse",
    "acceleration_mse",
    "kinetic_energy_mse",
    "temporal_frame_difference_error",
]
