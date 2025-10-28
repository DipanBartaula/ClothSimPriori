import torch
import numpy as np
import unittest
from torch.autograd import gradcheck

# Assuming the following paths are correct
from src.mpm_simulation import run_differentiable_mpm_simulation, MPMSimulationFunction
from src.models.physical_parameters import ClothPhysicalParameters

class TestMPMSimulation(unittest.TestCase):
    """
    Tests for the Differentiable MPM Simulation Function (JAX-PyTorch Bridge).
    Ensures correct shape, data types, and, crucially, gradient flow.
    """
    def setUp(self):
        # Configuration
        self.num_frames = 30
        self.num_particles = 100 
        self.device = torch.device("cpu") # Use CPU for stable JAX/NumPy conversion in tests

        # Initialize Trainable Parameters (PyTorch)
        # Using a reduced number of particles for faster testing
        self.params = ClothPhysicalParameters(num_initial_particles=self.num_particles).to(self.device)
        
        # Prepare Collider Motion (Input)
        # Shape: (num_frames, N_vertices, 3). Mock N_vertices=100
        self.collider_motion = torch.randn(self.num_frames, 100, 3, dtype=torch.float32, device=self.device)
        self.collider_motion.requires_grad_(False) # Collider motion is a fixed input, not a trained parameter

    def test_forward_pass_shape(self):
        """Tests that the forward pass returns the correct output shape."""
        
        # Run the simulation function
        output_pos = run_differentiable_mpm_simulation(self.collider_motion, self.params)
        
        # Output is the final position of all N particles (Shape: N, 3)
        expected_shape = (self.num_particles, 3)
        self.assertEqual(output_pos.shape, expected_shape)
        self.assertTrue(output_pos.requires_grad)

    def test_backward_pass_differentiability(self):
        """
        Tests the differentiability of the custom autograd function using gradcheck.
        This verifies that the gradients computed by the JAX VJP match the
        numerical gradients calculated by PyTorch.
        """
        print("\n--- Running Gradcheck for MPMSimulationFunction ---")
        
        # Convert all learned PyTorch parameters into a tuple of tensors for gradcheck
        # These are the inputs that require gradients.
        inputs = (
            self.params.initial_particle_offset,
            self.params.lame_lambda,
            self.params.lame_mu,
            self.params.mass_density,
            self.params.fiber_direction,
            self.collider_motion.detach().requires_grad_(False) # Non-learnable input
        )
        
        # The MPMSimulationFunction.apply method must be used with gradcheck
        # Setting eps (epsilon) higher is sometimes necessary due to the non-smoothness
        # or numerical stability of complex physical simulations.
        try:
            # We use `MPMSimulationFunction.apply` directly for the gradcheck
            test_passed = gradcheck(MPMSimulationFunction.apply, inputs, eps=1e-3, atol=1e-3, rtol=1e-3)
            self.assertTrue(test_passed, "Gradcheck failed: Analytic and numerical gradients do not match.")
            print("INFO: Gradcheck passed successfully.")
        except RuntimeError as e:
            # Catch common JAX/Torch conversion errors during gradcheck
            self.fail(f"Gradcheck failed due to RuntimeError (likely numerical instability or JAX issue): {e}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
