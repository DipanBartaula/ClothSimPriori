import torch
import numpy as np

from src.evaluation.metrics import chamfer_distance, psnr, ssim


def test_chamfer_identical():
    # identical point clouds -> zero distance
    pc = torch.randn(1, 128, 3)
    cd = chamfer_distance(pc, pc)
    assert torch.allclose(cd, torch.tensor(0.0), atol=1e-6)


def test_chamfer_shift():
    # shift one cloud by constant -> distance should equal shift^2
    pc1 = torch.zeros(1, 10, 3)
    pc2 = torch.zeros(1, 10, 3) + 0.5
    cd = chamfer_distance(pc1, pc2)
    # Each point distance squared = 0.5^2 * 3 dims = 0.25*3
    expected = torch.tensor(0.25 * 3)
    assert torch.allclose(cd, expected, atol=1e-6)


def test_psnr_identical():
    img = torch.rand(3, 64, 64)
    val = psnr(img, img)
    # identical images -> very large PSNR (division by eps handled)
    assert val > 60.0


def test_psnr_noise():
    img = torch.zeros(1, 1, 32, 32)
    noisy = img + 0.1
    val = psnr(img, noisy, data_range=1.0)
    # MSE = 0.01, PSNR = 20 dB
    assert torch.isclose(val, torch.tensor(20.0), atol=1e-2)


def test_ssim_identical():
    img = torch.rand(1, 3, 64, 64)
    s = ssim(img, img)
    assert torch.allclose(s, torch.tensor(1.0), atol=1e-6)


def test_ssim_degraded():
    img = torch.ones(1, 1, 32, 32) * 0.5
    noisy = img + 0.1 * torch.randn_like(img)
    s = ssim(img, noisy)
    assert (s < 1.0) and (s > 0.0)
