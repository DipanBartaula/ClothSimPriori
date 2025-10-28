import torch
import torch.nn as nn

class ClothPhysicalParameters(nn.Module):
    """
    A module to hold the learnable physical parameters of the cloth.
    
    We use torch.nn.Parameter so that they can be optimized.
    We use softplus to ensure parameters like stiffness and mass are positive.
    """
    def __init__(self):
        super().__init__()
        
        # We store the 'raw' parameters and expose the positive-constrained
        # parameters via properties.
        self._log_stiffness = nn.Parameter(torch.tensor(0.0))  # Example: Bending stiffness
        self._log_damping = nn.Parameter(torch.tensor(-2.0)) # Example: Damping coefficient
        self._raw_friction = nn.Parameter(torch.tensor(0.0))   # Example: Friction coefficient
        self._log_mass_density = nn.Parameter(torch.tensor(0.0)) # Example: Mass per area

    @property
    def stiffness(self):
        """Bending stiffness (must be positive)."""
        return torch.nn.functional.softplus(self._log_stiffness) + 1e-6

    @property
    def damping(self):
        """Damping coefficient (must be positive)."""
        return torch.nn.functional.softplus(self._log_damping) + 1e-6

    @property
    def friction(self):
        """Friction coefficient (e.g., 0 to 1)."""
        return torch.sigmoid(self._raw_friction)

    @property
    def mass_density(self):
        """Mass density (must be positive)."""
        return torch.nn.functional.softplus(self._log_mass_density) + 1e-6

    def get_all_params(self):
        """Returns a dictionary of the current parameter values."""
        return {
            'stiffness': self.stiffness,
            'damping': self.damping,
            'friction': self.friction,
            'mass_density': self.mass_density,
        }

    def forward(self):
        """Convenience method to return all parameters."""
        return self.get_all_params()