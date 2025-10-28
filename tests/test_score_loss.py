import torch
from src.losses.video_sds_loss import VideoSDSLoss

def test_sds_loss_forward():
    if not torch.cuda.is_available():
        print("Skipping SDS test, no CUDA.")
        return
        
    device = torch.device("cuda")
    
    # 1. Create arbitrary video tensor
    batch_size = 1
    num_frames = 16
    channels = 3
    height = 64 # Use small res for test
    width = 64
    
    # This tensor MUST require gradients
    video = torch.rand(
        batch_size, num_frames, channels, height, width,
        device=device,
        requires_grad=True
    )
    
    # 2. Initialize loss
    sds_loss_fn = VideoSDSLoss(device, text_prompt="test prompt")
    
    # 3. Calculate loss
    loss = sds_loss_fn(video)
    
    # 4. Check output
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 # Should be a scalar
    assert loss.item() >= 0
    
    # 5. Check backward pass
    loss.backward()
    assert video.grad is not None
    assert video.grad.shape == video.shape
    assert video.grad.norm() > 0
    print("VideoSDSLoss test passed.")