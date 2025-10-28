import torch
import torch.nn as nn
import torch.nn.functional as F

class ClothPhysicalParameters(nn.Module):
    """
    Trainable parameters for the cloth simulation, aligned with MPM-Avatar paper.
    This includes initial state optimization and anisotropic material properties.
    """
    def __init__(self, num_initial_particles=6890 * 8): # Typically many particles per vertex
        super().__init__()
        
        # 1. Initial State Optimization (Optimized particle positions/offsets)
        # This optimizes small offsets from the canonical cloth mesh/template.
        self.initial_particle_offset = nn.Parameter(
            torch.randn(num_initial_particles, 3) * 1e-4, 
            requires_grad=True
        )
        
        # 2. Anisotropic Physics Parameters (Lame parameters and fiber direction)
        
        # Mass Density (rho)
        self.mass_density = nn.Parameter(torch.tensor([1200.0]), requires_grad=True) # kg/m^3
        
        # Lame parameters (Lambda and Mu for Isotropic Elasticity Base)
        self.lame_lambda = nn.Parameter(torch.tensor([500.0]), requires_grad=True)
        self.lame_mu = nn.Parameter(torch.tensor([1500.0]), requires_grad=True)
        
        # Damping Coefficient
        self.damping = nn.Parameter(torch.tensor([0.05]), requires_grad=True)
        
        # Anisotropy / Fiber Direction (The vector defining the material's preferred stretch)
        # Normalized 3D vector
        initial_fiber = F.normalize(torch.tensor([0.0, 1.0, 0.0]), dim=0) 
        self.fiber_direction = nn.Parameter(initial_fiber, requires_grad=True)
        
        print(f"INFO: ClothPhysicalParameters initialized for {num_initial_particles} particles (MPM-Avatar style).")

    def forward(self):
        # Normalization and clamping might occur here during training
        return self

    def get_all_params(self):
        # Helper to retrieve current parameter values for logging/debugging
        return {
            "initial_particle_offset_norm": self.initial_particle_offset.norm().item(),
            "mass_density": self.mass_density.item(),
            "lame_lambda": self.lame_lambda.item(),
            "lame_mu": self.lame_mu.item(),
            "damping": self.damping.item(),
            "fiber_direction_y": self.fiber_direction[1].item()
        }