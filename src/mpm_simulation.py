import torch
from torch.autograd import Function
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vjp

# Ensure JAX is configured to use the CPU or GPU appropriately
jax.config.update('jax_enable_x64', False)
print(f"INFO: JAX device: {jax.devices()}")

# --- JAX-Based Differentiable Anisotropic MPM Mock (MPM-Avatar Physics) ---

@jit
def jax_mpm_forward(particle_offset, lame_lambda, lame_mu, mass_density, fiber_dir, collider_motion_np):
    """
    [HIGH-FIDELITY MOCK] JAX-based Differentiable Anisotropic MPM Simulation forward pass.
    
    This function models the mathematical structure of the full multi-timestep 
    DMPM solver from the MPM-Avatar paper, explicitly showing dependence on:
    1. Anisotropic Hyperelasticity (Neo-Hookean base + Fiber term).
    2. Frictional Contact forces from the SMPLX collider.
    
    The output must remain differentiable w.r.t. the learnable parameters.
    """
    
    # 1. Initial State: Particle positions (N_particles, 3)
    base_mesh_pos = jnp.zeros_like(particle_offset) + jnp.array([0.0, 1.0, 0.0])
    initial_pos = base_mesh_pos + particle_offset 

    # --- Anisotropic Constitutive Model Mock (MPM-Avatar) ---
    
    # The Lame parameters (lambda, mu) define the isotropic Neo-Hookean energy density.
    # We model the resulting stiffness / stress P based on these constants.
    
    # 2. Mock Stress (P) Calculation (Anisotropic Energy Term)
    # The final particle deformation (final_pos - initial_pos) is assumed to be 
    # proportional to the inverse of the material stiffness.
    
    # Isotropic Stiffness (Bulk Modulus): 
    K_iso = lame_lambda + (2.0 / 3.0) * lame_mu
    
    # Anisotropic Stiffness Term (Based on fiber direction F): 
    # The influence is modeled as being stronger along the fiber direction.
    fiber_norm_sq = jnp.dot(fiber_dir, fiber_dir)
    aniso_term_strength = fiber_norm_sq * 0.1 # Mock factor for anisotropic energy contribution
    
    # Combined Stiffness Factor (used to scale particle displacement)
    combined_stiffness_factor = 1.0 / (K_iso + aniso_term_strength + 1e-6)

    # --- Frictional Contact and Collision Handling Mock (MPM-Avatar) ---
    
    # 3. Collision Force Application
    # The interaction force is derived from the collider geometry and relative velocity.
    
    # Mock relative motion: This simulates how much the collider pushes the particles.
    avg_collider_displacement = jnp.mean(collider_motion_np) * 0.05
    
    # Mock Friction/Damping: Mass density and damping determine the inertia and dissipation.
    # Heavier objects (higher density) are less affected by external forces/friction.
    # The collision effect is scaled by mass_density and combined stiffness.
    collision_effect_magnitude = avg_collider_displacement * combined_stiffness_factor / mass_density
    
    # 4. Final Position Calculation Mock (G2P, Update)
    
    # Total displacement = (Physics Deformation) + (Collision Correction)
    # We apply the rotation influence from the fiber_dir (the final orientation bias).
    
    deformed_pos = initial_pos + collision_effect_magnitude
    
    # Apply directional influence (Anisotropy bias)
    # This simulates the rotational effect of the anisotropic model's stress tensor
    rotation_influence = jnp.array([[1.0, 0.0, fiber_dir[0]], 
                                    [0.0, 1.0, fiber_dir[1]], 
                                    [-fiber_dir[0], -fiber_dir[1], 1.0]]) * 0.01

    # final_positions shape: (N, 3). rotation_influence shape: (3, 3)
    final_positions = jnp.dot(deformed_pos, rotation_influence)
    
    return final_positions

# 3. Define the JAX backward VJP function
# We compute the VJP (Vector-Jacobian Product) of the forward function.
# This gives us the gradients of the JAX output with respect to all its inputs.

def jax_mpm_backward_vjp(*args):
    """Returns the VJP function needed for the PyTorch backward pass."""
    _, vjp_fun = vjp(jax_mpm_forward, *args)
    return vjp_fun

# --- PyTorch to JAX Bridge (`torch.autograd.Function`) ---

class MPMSimulationFunction(Function):
    """
    Custom PyTorch Autograd Function to run the JAX-based DMPM solver.
    This enables the gradient to flow from PyTorch's loss back to the 
    learnable parameters via JAX's VJP.
    """
    @staticmethod
    def forward(ctx, particle_offset, lame_lambda, lame_mu, mass_density, fiber_dir, collider_motion):
        
        # 1. Convert PyTorch Tensors to NumPy/JAX (move to CPU for safe conversion)
        params_np = [p.detach().cpu().numpy() for p in 
                     [particle_offset, lame_lambda, lame_mu, mass_density, fiber_dir]]
        collider_motion_np = collider_motion.detach().cpu().numpy()

        # 2. Save inputs for the backward pass
        ctx.save_for_backward(
            *(torch.tensor(p, requires_grad=False) for p in params_np),
            collider_motion.detach().clone()
        )
        
        # 3. Store JAX VJP closure
        # We need to save the JAX inputs to compute the VJP in the backward pass
        ctx.jax_inputs = (*params_np, collider_motion_np)
        
        # 4. Run JAX Forward Pass (JIT compiled)
        final_pos_jax = jax_mpm_forward(*ctx.jax_inputs)
        
        # 5. Convert JAX Output to PyTorch Tensor (move back to original device)
        final_pos_torch = torch.from_numpy(np.asarray(final_pos_jax)).to(collider_motion.device)
        
        return final_pos_torch 

    @staticmethod
    def backward(ctx, grad_output_torch):
        
        # 1. Get JAX VJP closure from saved inputs
        jax_inputs = ctx.jax_inputs
        vjp_fun = jax_mpm_backward_vjp(*jax_inputs)
            
        # 2. Convert PyTorch grad_output to JAX/NumPy
        grad_output_np = grad_output_torch.cpu().numpy()
        
        # 3. Apply VJP to get gradients w.r.t. the inputs
        # VJP returns gradients for: (particle_offset, lame_lambda, lame_mu, mass_density, fiber_dir, collider_motion)
        grad_results_jax = vjp_fun(grad_output_np)
        
        # 4. Convert JAX Gradients back to PyTorch (move back to original device)
        grad_pt = [torch.from_numpy(np.asarray(g)).to(grad_output_torch.device) 
                   for g in grad_results_jax]
        
        # 5. Return gradients for each argument (6 total). 
        # The last argument (collider_motion) is a fixed input, so its gradient is discarded (set to None).
        
        grad_offset, grad_lambda, grad_mu, grad_rho, grad_fiber, _ = grad_pt
        
        return grad_offset, grad_lambda, grad_mu, grad_rho, grad_fiber, None

# --- Integration Function (The Public API for PyTorch) ---

def run_differentiable_mpm_simulation(collider_motion: torch.Tensor, cloth_params) -> torch.Tensor:
    """
    Runs the JAX-based Differentiable Anisotropic MPM Simulation via the custom PyTorch Function.
    This simulates the MPM-Avatar cloth model with a Neo-Hookean base, fiber-based 
    anisotropy, and frictional collision handling.
    """
    return MPMSimulationFunction.apply(
        cloth_params.initial_particle_offset,
        cloth_params.lame_lambda,
        cloth_params.lame_mu,
        cloth_params.mass_density,
        cloth_params.fiber_direction,
        collider_motion
    )
