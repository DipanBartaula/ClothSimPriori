import torch
from src.smpl_processing import get_smpl_motion_from_image

def test_smpl_processing_pipeline():
    # Mock input
    image_path = "fake/image.png"
    motion_prompt = "a person walking"
    
    # Run placeholder function
    motion = get_smpl_motion_from_image(image_path, motion_prompt)
    
    # Check output
    assert isinstance(motion, torch.Tensor)
    assert motion.ndim == 3 # (frames, vertices, 3)
    assert motion.shape[0] > 1 # Should have multiple frames
    assert motion.shape[1] == 6890 # Standard SMPL vertex count
    assert motion.shape[2] == 3
    assert motion.is_cuda # Should be on GPU
    print("smpl_processing test passed.")