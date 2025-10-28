import torch
from src.pose_generation import MultimodalLLMPoseGenerator

def test_pose_generator():
    num_frames = 10
    pose_dim = 165
    
    generator = MultimodalLLMPoseGenerator(
        smplx_pose_dim=pose_dim, 
        num_frames=num_frames
    )
    
    # Run placeholder function
    poses = generator("fake/image.png", "a person waving")
    
    # Check output
    assert isinstance(poses, torch.Tensor)
    assert poses.ndim == 2 # (frames, pose_dim)
    assert poses.shape[0] == num_frames
    assert poses.shape[1] == pose_dim
    print("MultimodalLLMPoseGenerator test passed.")