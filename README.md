# ClothSimPriori — Short guide

This is a compact, interpretable README with a small block diagram and the essential equations used by the MPM-style differentiable cloth pipeline in this repo.

## Very short pipeline (block diagram)

Simple ASCII block diagram showing data + gradient flow:

```
    SMPL poses / motion    Precomputed 3D Gaussians
	    |                       |
	    v                       v
    +-------------+           +-------------+
    | Pose & Body |           | Gaussian    |
    | preprocessing|          | assets (pre)
    +-------------+           +-------------+
	    |                       |
	    v                       v
	 (optional)            +-----------+   Surface   +-----------+
	 refine /             | Simulator | --> Extract  | Renderer  | --> Image
	 non-trainable        |  (MPM)    |              | (diff)    |
	 assets               +-----------+              +-----------+
		 \                 ^  |
		  \_________________|  |  losses (image, feat, phys)
				     |  v
			      trainable params (optimized during training)
			      (mu, lambda, rest-shape)

Notes:
- Gaussian assets are produced by a separate single-image->3D pipeline
  (e.g., Gaussian Splatting / Prolific-Dreamer style refinement) before
  training. They are loaded as fixed colliders during training and are
  NOT optimized while tuning the physical parameters.
```

Gradients: image loss -> renderer -> surface -> particles -> simulator -> trainable params

## Core equations (compact)

Notation: p = particle index, i = grid node, w_ip = interpolation weight, m_p mass, V_p volume, x_p position, v_p velocity, F_p deformation gradient, dt timestep.

- P2G (mass & momentum):
	m_i = \sum_p w_{ip} m_p

	p_i = \sum_p w_{ip} m_p v_p

- Stress (compressible neo-Hookean, compact):
	\Psi(F) = \frac{\mu}{2}(\|F\|^2 - d) - \mu \log J + \frac{\lambda}{2}(\log J)^2, \quad J=\det F

	P(F) = \partial\Psi/\partial F = \mu(F - F^{-T}) + \lambda \log J\; F^{-T}

- Grid forces from particles:
	f_i = -\sum_p V_p P_p F_p^T \nabla w_{ip}

- Grid velocity update:
	v_i += dt * (f_i / m_i + g)

- G2P and F update:
	v_p^{n+1} = \sum_i w_{ip} v_i^{n+1}

	x_p^{n+1} = x_p^n + dt * v_p^{n+1}

	F_p^{n+1} = (I + dt \; (\nabla v)_p) F_p^n

Compact gradient (chain-rule) from image loss L to params θ:
	dL/dθ = dL/dI * dI/dS * dS/dX_T * dX_T/dθ

where X_T is final particle state after T timesteps and S is surface extraction.

## What is trainable (short)

- Physical parameters: mu, lambda (stiffness), damping, friction (see `models/physical_parameters.py`).
- Rest-shape offsets / per-particle priors.
- Note: Gaussian assets used to create collider geometry are assumed precomputed
	by a separate pipeline and are treated as fixed during training. Any neural
	modules used to create those assets (the single-image->3D refinement) are
	trained/ran offline and are not part of the training optimizer in
	`train.py`.

## Short file map

- `train.py` — training entry point.
- `inference.py` — evaluation / render pipeline.
- `mpm_simulation.py` — MPM steps (P2G, grid update, G2P, F update).
- `differentiable_collision.py` — collision handling.
- `differentiable_renderer.py` — simple differentiable renderer used for image losses.
- `models/physical_parameters.py` — physical parameter definitions.
- `gaussian_refinement.py` — optional learned refinement.
- `losses/` — image/feature/physics losses (JEPA, video SDS, etc.).

## Quick run

```powershell
python train.py   # run training
python inference.py   # run a demo inference
```

If you want, I can (pick one):
- add a PNG/SVG block diagram to the repo and reference it in this README,
- expand any equation into a derivation with a small figure, or
- generate a one-step runnable example script that uses `mpm_simulation.py` and shows a tiny visualization.


