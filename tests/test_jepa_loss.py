import torch
from src.losses.jepa_surprise_loss import JepaSurpriseLoss

def test_jepa_loss_forward():
    if not torch.cuda.is_available():
        print("Skipping JEPA test, no CUDA.")
        return
        
    device = torch.device("cuda")
    
    # 1. Create arbitrary video tensor
    batch_size = 2
    num_frames = 16 # Must be >= 2
    channels = 3
    height = 64
    width = 64
    
    video = torch.rand(
        batch_size, num_frames, channels, height, width,
        device=device,
        requires_grad=True
    )
    
    # 2. Initialize loss
    jepa_loss_fn = JepaSurpriseLoss(device)
    
    # 3. Calculate loss
    loss = jepa_loss_fn(video)
    
    # 4. Check output
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 # Scalar
    assert loss.item() >= 0
    
    # 5. Check backward pass
    loss.backward()
    assert video.grad is not None
    assert video.grad.shape == video.shape
    assert video.grad.norm() > 0
    print("JepaSurpriseLoss test passed.")