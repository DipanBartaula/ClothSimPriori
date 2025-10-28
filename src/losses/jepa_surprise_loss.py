import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Mock Model Placeholder ---
# In a real scenario, you would import the actual JEPA2 model.
class MockJEPA2Model(nn.Module):
    """
    A mock JEPA-style model.
    It has an 'encoder' and a 'predictor'.
    """
    def __init__(self, embed_dim=256, channels=3):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Encoder: Compresses a patch
        self.encoder = nn.Sequential(
            nn.Conv3d(channels, embed_dim // 4, 3, 2, 1), # (B, C, T, H, W) -> (B, D/4, T/2, H/2, W/2)
            nn.ReLU(),
            nn.Conv3d(embed_dim // 4, embed_dim, 3, 2, 1), # -> (B, D, T/4, H/4, W/4)
            nn.AdaptiveAvgPool3d((1, 1, 1)) # -> (B, D, 1, 1, 1)
        )
        
        # 2. Predictor: Predicts future embedding from context embedding
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        print("Using MOCK JEPA2 model.")

    def encode(self, video_clip):
        """Encodes a video clip into a latent representation."""
        # Input shape: (B, T, C, H, W)
        # Model expects: (B, C, T, H, W)
        video_clip = video_clip.permute(0, 2, 1, 3, 4)
        embedding = self.encoder(video_clip) # (B, D, 1, 1, 1)
        return embedding.squeeze() # (B, D)

    def predict(self, context_embedding):
        """Predicts the target embedding from the context embedding."""
        return self.predictor(context_embedding)

# -----------------------------

class JepaSurpriseLoss(nn.Module):
    """
    Calculates "surprise" using a JEPA model.
    
    "Surprise" is defined as the prediction error of the JEPA model.
    We aim to *minimize* this error, making the simulation look "plausible"
    and "predictable" to the pre-trained world model.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # Load the pre-trained JEPA model
        # In real use: self.model = load_jepa_model("path/to/jepa2_weights")
        self.model = MockJEPA2Model().to(device)
        self.model.eval() # Freeze the model
        print("JEPA Surprise Loss Initialized.")

    def forward(self, rendered_video):
        """
        Calculates the JEPA prediction loss.
        
        Args:
            rendered_video (torch.Tensor): The differentiably rendered video.
                                           Shape: (B, T, C, H, W)
        """
        
        # We need to split the video into context and target.
        # Let's use the first half as context and the second half as target.
        total_frames = rendered_video.shape[1]
        if total_frames < 2:
            raise ValueError("Video must have at least 2 frames for JEPA loss.")
            
        split_point = total_frames // 2
        
        context_clip = rendered_video[:, :split_point, ...]
        target_clip = rendered_video[:, split_point:, ...]
        
        # We must not backpropagate through the target encoder.
        with torch.no_grad():
            z_target = self.model.encode(target_clip)
        
        # We *do* backpropagate through the context encoder and predictor.
        z_context = self.model.encode(context_clip)
        z_pred = self.model.predict(z_context)
        
        # The "surprise" loss is the MSE between the prediction and the target.
        # We want to MINIMIZE this, making the video plausible.
        loss = F.mse_loss(z_pred, z_target)
        
        return loss