import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class VisualEncoder(nn.Module):
    """Wrapper around pretrained CLIP ViT-L/14 for feature extraction."""

    def __init__(
        self, model_name: str = "openai/clip-vit-large-patch14", device: str = None
    ):
        super().__init__()
        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"[MODEL] Loading CLIP model: {model_name} onto {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

        # Freeze all CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        print("[MODEL] CLIP ViT-L/14 loaded and frozen successfully.")

    @torch.no_grad()
    def forward(self, images: list) -> torch.Tensor:
        """Extracts normalized visual embeddings for a batch of PIL images.

        Output shape: [Batch_Size, 768]
        """
        inputs = self.processor(images=images, return_tensors="pt").to(
            self.device
        )

        # Extract features using CLIP's vision backbone
        image_outputs = self.model.get_image_features(**inputs)

        # Handle both tensor outputs and Hugging Face ModelOutput objects
        if hasattr(image_outputs, "image_embeds"):
            image_features = image_outputs.image_embeds
        elif hasattr(image_outputs, "pooler_output"):
            image_features = image_outputs.pooler_output
        elif isinstance(image_outputs, torch.Tensor):
            image_features = image_outputs
        else:
            image_features = image_outputs[0]

        # Normalize embeddings to unit length (L2 norm)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        return image_features


if __name__ == "__main__":
    print("=" * 65)
    print("[TEST] Initializing and validating VisualEncoder...")
    print("=" * 65)

    encoder = VisualEncoder()

    # Create dummy RGB image for testing (224x224)
    dummy_img = Image.new("RGB", (224, 224), color=(120, 150, 200))

    # Extract embedding
    embeddings = encoder([dummy_img])

    print("\n" + "=" * 65)
    print(f"[TEST SUCCESSFUL]")
    print(f"Extracted Embedding Shape : {embeddings.shape}")
    print(f"Embedding Vector Dtype    : {embeddings.dtype}")
    print(f"Vector L2-Norm (Expected ~1.0): {embeddings.norm(dim=-1).item():.4f}")
    print("=" * 65)