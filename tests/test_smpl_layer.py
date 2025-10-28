import torch
from src.models.smpl_layer import MockSMPLLayer

def test_mock_smpl_layer():
    num_frames = 10
    pose_dim = 165
    num_vertices = 6890
    
    layer = MockSMPLLayer(num_vertices=num_vertices)
    
    # Create arbitrary pose tensor
    poses = torch.rand(num_frames, pose_dim)
    
    # Run layer
    vertices = layer(poses)
    
    # Check output
    assert isinstance(vertices, torch.Tensor)
    assert vertices.ndim == 3 # (frames, vertices, 3)
    assert vertices.shape[0] == num_frames
    assert vertices.shape[1] == num_vertices
    assert vertices.shape[2] == 3
    print("MockSMPLLayer test passed.")