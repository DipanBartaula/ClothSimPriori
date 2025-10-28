import torch
import torch.optim as optim
import wandb
import numpy as np
from collections import deque

from src.models.physical_parameters import ClothPhysicalParameters
from src.losses.video_sds_loss import VideoSDSLoss
from src.losses.jepa_surprise_loss import JepaSurpriseLoss

# Import placeholder functions
from src.smpl_processing import get_smpl_motion_from_image
from src.gaussian_refinement import refine_smplx_with_3d_gaussians
from src.mpm_simulation import run_differentiable_mpm_simulation
from src.differentiable_renderer import render_simulation_to_video

# --- Config ---
CONFIG = {
    "epochs": 50,
    "iterations_per_epoch": 500,
    "lr": 1e-2,
    "wandb_project": "cloth_param_optimization",
    "log_iter_interval": 1,
    "log_stats_interval": 100, # Log mean/std every 100 iters
    "log_video_interval": 250,
    "video_frames": 30,
    "video_height": 256,
    "video_width": 256,
    "sds_prompt": "photorealistic video of a silk cloth falling onto a person",
    "input_image": "data/input_person.jpg",
}

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize WandB
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    
    # 2. Initialize Model (The learnable parameters)
    physical_params = ClothPhysicalParameters().to(device)
    
    # 3. Initialize Optimizer
    optimizer = optim.Adam(physical_params.parameters(), lr=CONFIG["lr"])
    
    # 4. Initialize Loss Functions
    sds_loss_fn = VideoSDSLoss(device, text_prompt=CONFIG["sds_prompt"])
    jepa_loss_fn = JepaSurpriseLoss(device)
    
    # 5. Load "prior" data (placeholders)
    # This is assumed to be done once
    raw_smpl_motion = get_smpl_motion_from_image(
        CONFIG["input_image"],
        CONFIG["motion_prompt"] # Pass the prompt
    ).to(device)
    collider_motion = refine_smplx_with_3d_gaussians(raw_smpl_motion)
    # Buffers for logging stats
    sds_loss_window = deque(maxlen=CONFIG["log_stats_interval"])
    jepa_loss_window = deque(maxlen=CONFIG["log_stats_interval"])

    print("INFO: Starting training loop...")
    global_step = 0
    for epoch in range(CONFIG["epochs"]):
        for iteration in range(CONFIG["iterations_per_epoch"]):
            
            # --- Main Optimization Step ---
            
            # 1. Run Differentiable Simulation
            # This is the "forward pass" of our "generator"
            # It must be differentiable w.r.t. physical_params
            sim_output = run_differentiable_mpm_simulation(
                collider_motion, 
                physical_params
            )
            
            # 2. Differentiable Rendering
            # Renders the simulation output to a video
            rendered_video = render_simulation_to_video(
                sim_output, 
                CONFIG["video_frames"], 
                CONFIG["video_height"], 
                CONFIG["video_width"]
            )
            
            # --- Loss Calculation & GradNorm Balancing ---
            
            # We need to compute gradients for each loss *separately*
            # to find their norms.
            
            optimizer.zero_grad()
            
            # --- SDS Loss ---
            loss_sds = sds_loss_fn(rendered_video)
            # Compute gradients *only* for SDS
            loss_sds.backward(retain_graph=True) 
            
            # Get the norm of the SDS gradient
            # We clip just in case, but float('inf') means no clipping
            grad_sds_norm = torch.nn.utils.clip_grad_norm_(
                physical_params.parameters(), float('inf')
            )
            
            # Clear gradients to compute the next one
            optimizer.zero_grad()
            
            # --- JEPA Loss ---
            loss_jepa = jepa_loss_fn(rendered_video)
            # Compute gradients *only* for JEPA
            loss_jepa.backward(retain_graph=True)
            
            grad_jepa_norm = torch.nn.utils.clip_grad_norm_(
                physical_params.parameters(), float('inf')
            )
            
            # Clear gradients before the final backward pass
            optimizer.zero_grad()
            
            # --- GradNorm Scaling ---
            # We scale each loss inversely by its gradient norm.
            # Add epsilon for numerical stability.
            scale_sds = 1.0 / (grad_sds_norm + 1e-8)
            scale_jepa = 1.0 / (grad_jepa_norm + 1e-8)
            
            # Optional: Normalize the scales so they sum to 1
            # total_scale = scale_sds + scale_jepa
            # scale_sds = scale_sds / total_scale
            # scale_jepa = scale_jepa / total_scale

            # --- Final Backward Pass ---
            # Now we apply the balanced loss
            loss_total = (scale_sds.detach() * loss_sds) + \
                         (scale_jepa.detach() * loss_jepa)
                         
            loss_total.backward()
            
            # --- Optimizer Step ---
            optimizer.step()

            # --- Logging ---
            sds_loss_window.append(loss_sds.item())
            jepa_loss_window.append(loss_jepa.item())
            
            log_data = {
                "epoch": epoch,
                "iteration": iteration,
                "total_loss": loss_total.item(),
                "sds_loss": loss_sds.item(),
                "jepa_loss": loss_jepa.item(),
                "grad_norm_sds": grad_sds_norm.item(),
                "grad_norm_jepa": grad_jepa_norm.item(),
                "scale_sds": scale_sds.item(),
                "scale_jepa": scale_jepa.item(),
            }
            
            # Log current parameter values
            params_dict = physical_params.get_all_params()
            for key, val in params_dict.items():
                log_data[f"param_{key}"] = val.item()

            if global_step % CONFIG["log_iter_interval"] == 0:
                wandb.log(log_data, step=global_step)
            
            if global_step % CONFIG["log_stats_interval"] == 0:
                sds_mean = np.mean(sds_loss_window)
                sds_std = np.std(sds_loss_window)
                jepa_mean = np.mean(jepa_loss_window)
                jepa_std = np.std(jepa_loss_window)
                
                wandb.log({
                    "sds_loss_mean": sds_mean,
                    "sds_loss_std": sds_std,
                    "jepa_loss_mean": jepa_mean,
                    "jepa_loss_std": jepa_std,
                }, step=global_step)

            if global_step % CONFIG["log_video_interval"] == 0:
                print(f"Step {global_step}: Logging video...")
                # Format for wandb.Video: (T, C, H, W) and [0, 255]
                video_to_log = rendered_video[0].detach().cpu().permute(0, 2, 3, 1)
                video_to_log = (video_to_log * 255.0).to(torch.uint8)
                wandb.log(
                    {"output_video": wandb.Video(video_to_log, fps=10)},
                    step=global_step
                )
                
            global_step += 1

    print("INFO: Training finished.")
    # Save the final model
    model_path = "trained_cloth_params.pth"
    torch.save(physical_params.state_dict(), model_path)
    wandb.save(model_path)
    wandb.finish()


if __name__ == "__main__":
    train()