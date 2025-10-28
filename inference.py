import torch
import argparse
import imageio # For saving video

from src.models.physical_parameters import ClothPhysicalParameters

# Import placeholder functions
from src.smpl_processing import get_smpl_motion_from_image
from src.gaussian_refinement import refine_smplx_with_3d_gaussians
from src.mpm_simulation import run_differentiable_mpm_simulation

# For inference, we'd use a *non-differentiable*, high-quality renderer
# We'll just re-use the placeholder for this example.
from src.differentiable_renderer import render_simulation_to_video

def run_inference(weights_path, input_image, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"INFO: Loading learned parameters from {weights_path}")
    
    # 1. Load learned parameters
    physical_params = ClothPhysicalParameters().to(device)
    physical_params.load_state_dict(torch.load(weights_path, map_location=device))
    physical_params.eval() # Set to evaluation mode
    
    print("INFO: Learned parameters:")
    params_dict = physical_params.get_all_params()
    for key, val in params_dict.items():
        print(f"  {key}: {val.item():.4f}")

    # 2. Get SMPL motion
    print(f"INFO: Loading SMPL motion from {input_image}")
    raw_smpl_motion = get_smpl_motion_from_image(
        input_image, 
        motion_prompt
    ).to(device)
    collider_motion = refine_smplx_with_3d_gaussians(raw_smpl_motion)
    # 3. Run simulation with *learned* parameters
    # We don't need gradients for inference
    with torch.no_grad():
        print("INFO: Running final simulation with learned parameters...")
        # Note: We use the *differentiable* sim function here,
        # but in a real case, you might have a separate, faster
        # non-differentiable version for inference.
        sim_output = run_differentiable_mpm_simulation(
            collider_motion, 
            physical_params
        )
        
        # 4. Render the output
        # Here you would use a high-quality (non-differentiable) renderer
        # like Mitsuba, Blender, or a game engine.
        print("INFO: Rendering high-quality output video...")
        rendered_video = render_simulation_to_video(
            sim_output, 
            num_frames=sim_output.shape[0], 
            height=512, 
            width=512
        ) # (B, T, C, H, W)

    # 5. Save the video
    print(f"INFO: Saving video to {output_path}")
    video_data = rendered_video[0].permute(0, 2, 3, 1).cpu().numpy() # (T, H, W, C)
    video_data = (video_data * 255).astype("uint8")
    
    imageio.mimsave(output_path, video_data, fps=30)
    print("INFO: Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", 
        type=str, 
        default="trained_cloth_params.pth",
        help="Path to the trained physical parameter weights."
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default="data/input_person.jpg",
        help="Path to the input image."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output_simulation.mp4",
        help="Path to save the output video."
    )
    args = parser.parse_args()
    
    run_inference(args.weights, args.image, args.output)