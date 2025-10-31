import math
import torch
import torch.nn.functional as F


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric Chamfer Distance between two point clouds.

    Args:
        pc1: (B, N, D) or (N, D) tensor
        pc2: (B, M, D) or (M, D) tensor

    Returns:
        Tensor of shape (B,) or scalar with the mean chamfer distance.
    """
    # Ensure batch dimension
    if pc1.dim() == 2:
        pc1 = pc1.unsqueeze(0)
    if pc2.dim() == 2:
        pc2 = pc2.unsqueeze(0)

    assert pc1.dim() == 3 and pc2.dim() == 3, "Point clouds must be (B,N,D)"
    B, N, D = pc1.shape
    _, M, _ = pc2.shape

    # Compute pairwise squared distances: (x-y)^2 = x^2 + y^2 - 2xy
    pc1_sq = (pc1 ** 2).sum(dim=2, keepdim=True)  # (B,N,1)
    pc2_sq = (pc2 ** 2).sum(dim=2, keepdim=True)  # (B,M,1)

    # (B, N, M) distances
    # Use broadcasting carefully
    dists = pc1_sq + pc2_sq.transpose(1, 2) - 2 * (pc1 @ pc2.transpose(1, 2))
    # clamp small negatives from numerical error
    dists = torch.clamp(dists, min=0.0)

    # For each point in pc1 find nearest in pc2, and vice versa
    mins1, _ = torch.min(dists, dim=2)  # (B, N)
    mins2, _ = torch.min(dists, dim=1)  # (B, M)

    # Mean over points
    cd = mins1.mean(dim=1) + mins2.mean(dim=1)  # (B,)
    return cd if cd.numel() > 1 else cd.squeeze(0)


def psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Compute Peak Signal to Noise Ratio between two images.

    Args:
        img1, img2: tensors of shape (B,C,H,W) or (C,H,W) or (H,W). Values in [0, data_range].
        data_range: maximum possible pixel value (default 1.0)

    Returns:
        PSNR in dB as tensor scalar or per-batch tensor.
    """
    # Ensure batch
    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    mse = F.mse_loss(img1, img2, reduction="none")
    mse = mse.view(mse.shape[0], -1).mean(dim=1)
    # Avoid division by zero
    eps = 1e-10
    psnr_val = 10.0 * torch.log10((data_range ** 2) / (mse + eps))
    return psnr_val if psnr_val.numel() > 1 else psnr_val.squeeze(0)


def _gaussian_window(window_size: int, sigma: float, channel: int, device=None):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = g.unsqueeze(1) @ g.unsqueeze(0)  # outer product -> 2D gaussian
    w = w.expand(channel, 1, window_size, window_size)
    return w


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> torch.Tensor:
    """
    Compute structural similarity (SSIM) index between two images.

    This is a lightweight PyTorch implementation suitable for tests. It
    computes the mean SSIM over channels and spatial dimensions.

    Args:
        img1, img2: tensors (B,C,H,W) or (C,H,W) or (H,W)
        window_size: Gaussian window size
        data_range: maximum pixel value

    Returns:
        SSIM value in [0,1], per-batch tensor or scalar.
    """
    # Normalize inputs to float
    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    B, C, H, W = img1.shape
    device = img1.device

    sigma = 1.5
    window = _gaussian_window(window_size, sigma, C, device=device)

    # pad for 'same' conv
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, groups=C, padding=pad)
    mu2 = F.conv2d(img2, window, groups=C, padding=pad)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=C, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=C, padding=pad) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=C, padding=pad) - mu1_mu2

    # Constants to stabilize the division
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Mean over spatial dimensions and channels
    ssim_val = ssim_map.view(B, -1).mean(dim=1)
    return ssim_val if ssim_val.numel() > 1 else ssim_val.squeeze(0)


__all__ = ["chamfer_distance", "psnr", "ssim"]
