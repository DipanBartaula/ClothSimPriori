# ClothSimPriori — Differentiable Cloth Simulation & Learning

This repository implements a differentiable cloth simulation and learning pipeline in the spirit of modern differentiable MPM systems (e.g. MPMAvatar-style systems). The code combines a Material Point Method (MPM) simulator, differentiable collision and rendering, and data/loss functions that allow optimizing physical parameters and neural refinement modules end-to-end from image or video losses.

This README summarizes the mathematical model, the gradient flow (how gradients reach trainable parameters), training recipe, and a short description of the most important files and modules in this repository.

## High-level overview

- Input: posed characters / body motion (SMPL), initial cloth geometry and material priors.
- Differentiable simulator: MPM-based cloth dynamics stepper with collision handling.
- Differentiable renderer: turns simulated cloth surface into images for image-space losses.
- Losses: combination of image-space, feature-space (e.g., JEPA-style surprise) and physics priors.
- Optimizer: minimizes losses w.r.t. trainable parameters (material parameters, rest-shape refinements, and optional neural modules).

The pipeline is designed so gradients computed from image/feature losses backpropagate through the renderer and simulator to update trainable physical and neural parameters.

## Mathematical model (MPM-style cloth)

Notation (discrete particle set):
- p indexes particles (material points).
- i indexes grid nodes.
- x_p, v_p: particle position and velocity.
- m_p: particle mass; V_p: particle volume (or reference volume V0_p).
- F_p: deformation gradient for particle p.
- dt: timestep.

1) Particle-to-grid (P2G)

We transfer mass and momentum to an Eulerian grid using interpolation weights w_{ip} (e.g., B-spline / cubic weights):

\[
m_i = \sum_p w_{ip} m_p
\]
\[
\mathbf{p}_i = \sum_p w_{ip} m_p \mathbf{v}_p
\]
\[
\mathbf{v}_i = \frac{\mathbf{p}_i}{m_i} \quad (\text{if } m_i>0)
\]

2) Compute stresses and grid forces

For hyperelastic cloth constitutive models (Neo-Hookean-like), define a strain energy density \(\Psi(F)\). A commonly used compressible Neo-Hookean energy is:

\[
\Psi(F) = \frac{\mu}{2}\left(\|F\|^2 - d\right) - \mu \log J + \frac{\lambda}{2}(\log J)^2,
\]
where \(J = \det F\), and \(\mu, \lambda\) are Lamé parameters (material stiffnesses), and \(d\) is spatial dimension (2 or 3).

The first Piola–Kirchhoff stress is

\[
P(F) = \frac{\partial \Psi}{\partial F} = \mu(F - F^{-T}) + \lambda \log J \, F^{-T}.
\]

Particle stress gives force contributions to the grid via the discrete divergence of stress (using the gradients of interpolation weights):

\[
\mathbf{f}_i = -\sum_p V_p P_p F_p^T \nabla w_{ip}.
\]

Grid velocities are updated with external forces (gravity), internal elastic forces, and boundary/collision impulses:

\[
\mathbf{v}_i \leftarrow \mathbf{v}_i + \frac{dt}{m_i} \mathbf{f}_i + dt \, \mathbf{g}
\]

3) Grid-to-particle (G2P)

Update particle velocities and positions by gathering velocities from the grid (optionally using APIC/PIC blending):

\[
\mathbf{v}_p^{n+1} = \sum_i w_{ip} \mathbf{v}_i^{n+1}
\]
\[
\mathbf{x}_p^{n+1} = \mathbf{x}_p^{n} + dt \; \mathbf{v}_p^{n+1}
\]

And update deformation gradient F_p using velocity gradients (\nabla v):

\[
F_p^{n+1} = (I + dt \; (\nabla v)_p) F_p^{n}
\]

4) Constitutive and collision handling

- Collision handling modifies grid velocities with boundary conditions or differentiable collision impulses implemented in `differentiable_collision.py`.
- Cloth-specific constraints (e.g., stretch-bending terms) can be built into the particle stress computation or as soft constraints added as forces.

## Trainable parameters and gradient flow

Trainable parameters (examples present or implied in this repo):
- Material parameters: \(\mu, \lambda\), damping coefficients, friction coefficients (see `models/physical_parameters.py`).
- Rest-shape / per-particle rest lengths or offsets (small adjustments to initial particle positions or rest volumes).
- Neural refinement parameters: weights of any refinement networks (e.g., `gaussian_refinement.py` or other learned modules that post-process sim output).
- Renderer parameters if differentiable renderer has learnable components.

Gradient flow (conceptual ASCII diagram):

Simulator step is denoted S(·; θ_sim) where θ_sim are simulator/trainable phys params.

\[
\begin{array}{c}
	ext{Trainable parameters } \theta = \{\theta_{phys}, \theta_{nn}, \theta_{rest}\}\n+\\
\downarrow
\\
	ext{Simulator (MPM)}:\; X_T = S(X_0, M, \theta)
\\
\downarrow
\\
	ext{Surface extraction / rasterization} \; S \rightarrow \mathcal{S}\n+\\
\downarrow
\\
	ext{Differentiable renderer } R(\mathcal{S}) \rightarrow I
\\
\downarrow
\\
	ext{Loss } L(I, I_{gt})
\end{array}
\]

Backpropagation chain (by chain rule):

\[
\frac{dL}{d\theta} = \frac{dL}{dI} \frac{dI}{d\mathcal{S}} \frac{d\mathcal{S}}{dX_T} \frac{dX_T}{d\theta}.
\]

Expanding the last term through time-unrolled simulation steps for t=0..T gives gradients flowing through each time-step's P2G / grid update / G2P operations down to the parameters that affect stress/forces and rest-shape.

Intuitively:
- Image loss gradients dL/dI influence surface geometry via renderer partials dI/dS.
- dS/dX maps surface changes to particle position changes (mesh/point extraction / splatting gradients).
- dX/dθ captures sensitivity of particle trajectories to material parameters and rest-shape (through stresses and forces in P2G/G2P).

Note: numerical stability of gradient propagation is enhanced by using stable constitutive models, clamping of singular values when computing F^{-T} and careful time-step selection.

## Losses used in training

- Image-space L2 / perceptual losses: compare rendered cloth to ground-truth images or features.
- JEPA-style surprise loss (see `losses/jepa_surprise_loss.py`): used for self-supervised feature matching.
- Video SDS / score-based supervisory terms (see `losses/video_sds_loss.py`) for matching dynamics in feature space.
- Physical regularizers: penalty terms on excessive stretch, large strains, or deviation from prior material parameters.

Total loss often has the form:

\[
L = \lambda_{img} L_{img} + \lambda_{feat} L_{feat} + \lambda_{phys} L_{phys}
\]

where the lambdas weigh different objectives.

## Training recipe (typical)

1. Prepare a batch of poses / motions (SMPL) using `pose_generation.py` and `smpl_processing.py`.
2. Initialize cloth particle state (positions, masses, F) and material priors from `models/physical_parameters.py`.
3. Rollout simulator `mpm_simulation.py` for T timesteps producing particle histories.
4. Extract surfaces and render frames with `differentiable_renderer.py`.
5. Compute losses in `losses/` and backpropagate through renderer and simulator to update trainable parameters using an optimizer (Adam recommended).

Mini-batch tip: if simulation per example is heavy, accumulate gradients across few rollouts or use gradient checkpointing / shorter unrolls.

## Files and modules (quick map)

- `train.py` — training loop / experiment harness (entry point for training experiments).
- `inference.py` — inference / evaluation runner.
- `mpm_simulation.py` — core differentiable MPM-based simulator and time-stepping. Implements P2G, stress computation, grid update, G2P, and F updates.
- `differentiable_collision.py` — differentiable collision handling and boundary conditions.
- `differentiable_renderer.py` — lightweight differentiable renderer for producing image-space predictions from simulated cloth surface points/meshes.
- `gaussian_refinement.py` — neural/refinement module(s) that operate on particle/gaussian representations (optional trainable post-processing).
- `pose_generation.py`, `smpl_processing.py` — utilities to synthesize/prepare SMPL poses and convert them to per-frame transforms and collision proxies.
- `models/physical_parameters.py` — definitions of trainable physical parameters and convenience functions to create parameter tensors.
- `models/smpl_layer.py` — a utility layer for SMPL integrations.
- `losses/*.py` — loss implementations used in training (JEPA surprise, video SDS, etc.).
- `tests/*.py` — unit tests for simulation components, losses, and model pieces.

## Practical notes and tips

- Numerical stability: clamp singular values in SVD when computing F^{-T} or use stable neo-Hookean formulations.
- Time-step: choose dt small enough for stability; consider semi-implicit integrators for stiffer materials.
- Differentiable collisions: ensure collisions are implemented with sub-gradients (not discrete rewinds) so gradients remain useful.
- Regularization: add small physical priors to prevent degenerate learned parameters (e.g., negative stiffness).

## How to run (quick)

You can run training and inference via the provided scripts. On Windows PowerShell, a typical command is:

```powershell
python train.py
```

Or run inference:

```powershell
python inference.py
```

There are helper shell scripts (`run_training.sh`, `run_inference.sh`, `run_tests.sh`) for Unix-like shells; on Windows PowerShell invoke the corresponding Python scripts directly.

## Reproducibility & experiments

- Seed RNGs for reproducibility in `train.py`.
- Keep checkpoints of both simulator parameters and optimizer states.

## References and inspiration

This implementation is inspired by recent differentiable simulation and learning works that leverage MPM-style simulators to make physical parameters trainable from image/video losses. The math and pipeline follow standard MPM derivations and hyperelastic constitutive formulations.

## Contact / Contributing

If you find issues or want to contribute (fixes, new loss terms, renderer improvements), please open an issue or a PR against this repository.

---

End of README — updated with mathematical background, gradient flow, and module map.

