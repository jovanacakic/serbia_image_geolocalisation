"""
geoclip_adaptation/model.py — GeoCLIP adapted for direct GPS regression on
Serbia. Two adaptation strategies are supported so the ablation ladder
(see train_lora.py) can measure each design decision on our own data:

  - "full_unfreeze": directly unfreeze the last N ViT layers. This is the
    regime most at risk of catastrophic forgetting at standard LR, which we
    test rather than assume.
  - "lora": LoRA adapters on configurable attention projections/rank
    ({q,v} r=8 through {q,k,v,out} r=16), plus unfreezing
    GeoCLIP's own visual_projection and image-encoder MLP as a lightweight
    domain-adaptation bottleneck, matching their recipe exactly.

Either way, GeoCLIP's native gallery-retrieval mechanism (location_encoder,
gps_gallery) is discarded — see train_lora.py's docstring for why regression
was chosen over retrieval for a small-country target like Serbia.

Requires transformers<5.0 (pinned in requirements.txt). Verified locally:
GeoCLIP's own `ImageEncoder.forward` — with zero modifications from this
project — raises a TypeError under transformers>=5.0, because
CLIPModel.get_image_features started returning a BaseModelOutputWithPooling
object instead of a plain tensor. transformers==4.46.3 confirmed working.
"""
import torch
import torch.nn as nn
from geoclip import GeoCLIP
from peft import LoraConfig, get_peft_model

# GeoCLIP's CLIPModel has identically-named q_proj/k_proj/v_proj/out_proj
# modules in its text encoder too (unused — ImageEncoder.forward never calls
# it). This regex scopes LoRA injection to only the vision_model branch;
# verified empirically at targets=(q,k,v,out), r=16 to produce exactly
# 3,145,728 trainable LoRA parameters.
_PROJ_NAMES = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "out": "out_proj"}


def _vision_attn_pattern(targets):
    proj_alternation = "|".join(_PROJ_NAMES[t] for t in targets)
    return rf".*vision_model.*({proj_alternation})$"


# Serbia bounding box for the regression head's sigmoid output range.
# Derived from data/osv5m_serbia_metadata.csv's actual lat/lon extent
# (lat 42.2607-46.1322, lon 18.9240-22.8516), rounded outward with a margin
# so every real training point — and any TTA-shifted prediction — falls
# strictly inside the box rather than at its edge. NOT taken directly from
# the Serbia border polygon: that polygon's own bounding box (lat
# 42.3068-46.1832) is *tighter* than the real data on the southern edge,
# which would clip real points at the sigmoid boundary.
SERBIA_LAT_MIN, SERBIA_LAT_MAX = 42.1, 46.2
SERBIA_LON_MIN, SERBIA_LON_MAX = 18.8, 23.0


class GeoCLIPRegressor(nn.Module):
    def __init__(
        self,
        adaptation="lora",
        lora_targets=("q", "k", "v", "out"),
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        unfreeze_last_n_layers=2,
    ):
        super().__init__()
        assert adaptation in ("lora", "full_unfreeze")
        self.adaptation = adaptation

        geoclip = GeoCLIP(from_pretrained=True)
        # Only image_encoder is kept — location_encoder/gps_gallery/gps_queue
        # are GeoCLIP's retrieval machinery, unused by a regression model and
        # left to be garbage-collected with the rest of `geoclip`.
        self.image_encoder = geoclip.image_encoder
        clip = self.image_encoder.CLIP

        # Both strategies start fully frozen (GeoCLIP's own __init__ already
        # freezes all of self.CLIP; image_encoder.mlp was never frozen).
        if adaptation == "lora":
            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=_vision_attn_pattern(lora_targets),
                lora_dropout=lora_dropout, bias="none",
            )
            # get_peft_model freezes everything except the matched LoRA
            # layers by default — visual_projection needs an explicit
            # unfreeze below to match the recipe (trained alongside the
            # LoRA adapters as a "lightweight domain-adaptation bottleneck").
            self.image_encoder.CLIP = get_peft_model(clip, lora_cfg)
            self.image_encoder.CLIP.visual_projection.weight.requires_grad_(True)
        else:
            # "full_unfreeze": no LoRA at all — directly unfreeze the last
            # N vision transformer blocks plus visual_projection. Mirrors
            # their V1 (28.1M trainable / 6.4%, "2 ViT layers"), which their
            # paper reports produces catastrophic forgetting at LR=1e-4.
            vision_layers = clip.vision_model.encoder.layers
            for layer in vision_layers[-unfreeze_last_n_layers:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            clip.visual_projection.weight.requires_grad_(True)
        # image_encoder.mlp was never frozen in the first place — GeoCLIP's
        # own __init__ only freezes self.CLIP's parameters, not this MLP —
        # so no action needed here; verified trainable by default.

        self.regressor = nn.Sequential(
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 2), nn.Sigmoid(),
        )

        self.register_buffer(
            "bbox_min", torch.tensor([SERBIA_LAT_MIN, SERBIA_LON_MIN], dtype=torch.float32)
        )
        self.register_buffer(
            "bbox_max", torch.tensor([SERBIA_LAT_MAX, SERBIA_LON_MAX], dtype=torch.float32)
        )

    def preprocess_image(self, image):
        """Delegates to GeoCLIP's own CLIP image processor, so LoRA fine-tuning
        sees exactly the preprocessing pipeline the pretrained weights expect."""
        return self.image_encoder.preprocess_image(image)

    def forward(self, pixel_values):
        features = self.image_encoder(pixel_values)      # [B, 512]
        normalized = self.regressor(features)              # [B, 2] in [0, 1]
        coords = normalized * (self.bbox_max - self.bbox_min) + self.bbox_min
        return coords                                       # [B, 2] = (lat, lon)

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
