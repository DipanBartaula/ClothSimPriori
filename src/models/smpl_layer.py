import torch
import torch.nn as nn

class MockSMPLLayer(nn.Module):
    """
    (Placeholder)
    This class mocks the official SMPL/SMPLX model layer.
    
    The real layer would take pose parameters, shape parameters (betas),
    and global orientation/translation to produce vertices.
    
    See: https://github.com/vchoutas/smplx
    """
    def __init__(self, num_vertices=6890):
        super().__init__()
        self.num_vertices = num_vertices
        
        # Mock base mesh shape
        base_mesh = torch.randn(self.num_vertices, 3)
        self.register_buffer("base_mesh", F.normalize(base_mesh, dim=1))
        
        # Mock linear blend skinning (LBS) as a simple linear layer
        # This is a massive simplification.
        self.mock_lbs = nn.Linear(165, self.num_vertices * 3)
        print("Using MOCK SMPLLayer.")

    def forward(self, pose_params: torch.Tensor):
        """
        Converts pose parameters into mesh vertices.
        
        Args:
            pose_params (torch.Tensor): SMPLX poses.
                                        Shape: (num_frames, pose_dim)
        
        Returns:
            torch.Tensor: Mesh vertices.
                          Shape: (num_frames, num_vertices, 3)
        """
        
        # Mock LBS: apply poses to deform the base mesh
        deformations = self.mock_lbs(pose_params).reshape(
            pose_params.shape[0], self.num_vertices, 3
        )
        
        # Add deformation to base mesh
        # We add a small multiplier to keep deformations reasonable
        vertices = self.base_mesh.unsqueeze(0) + (deformations * 0.1)
        
        return vertices