import clip
import torch

def load_clip_model():
    """Load CLIP model and preprocessing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess
