"""
g3_adaptation/model.py — the geo-alignment component of G3 (Jia et al.,
NeurIPS 2024, arXiv:2405.14702), adapted to Serbia.

G3 has three stages:
  1. Geo-alignment      — contrastively align image / GPS (/ text) into one
                          embedding space. THIS FILE.
  2. Geo-diversification — an MLLM generates K x N candidate coordinates from
                          RAG prompts. Out of scope here: needs an external
                          multimodal LLM. We substitute the S retrieved
                          neighbours that G3's own candidate pool also
                          includes (see evaluate_g3.py).
  3. Geo-verification   — score candidates by cosine similarity between the
                          image embedding and each candidate's GPS embedding,
                          using the stage-1 encoders. Implemented, in
                          evaluate_g3.py.

G3's stage 1 explicitly follows GeoCLIP's design, which is also what the
GeoToken paper cites it for ("Encoder Pretraining via Geo-Alignment"). We
therefore reuse GeoCLIP's location-encoder architecture (equal-earth
projection -> random Fourier features at several scales -> MLP -> 512-d),
optionally starting from its pretrained weights.

The text modality is included (`use_text=True`, the default), built from
OSV5M's own reverse-geocoded place fields — see g3_adaptation/dataset.py for
what Serbia's hierarchy actually supports. Setting `use_text=False` reduces
stage 1 to image<->GPS alignment, which is exactly GeoCLIP's formulation, and
serves as the ablation isolating what the text modality contributes.

Deviations from the paper, forced by this project's data/compute:
  - Both CLIP towers are frozen by default (`--image_adaptation frozen_mlp`),
    training only the projection MLPs. Faithful to GeoCLIP's own pretraining,
    which freezes CLIP entirely. LoRA on the vision tower is available for a
    capacity comparison mirroring the geoclip_adaptation ladder; the text
    tower stays frozen either way, which lets its outputs be precomputed once
    per unique description instead of re-encoded every batch.
  - A flatter text hierarchy than G3's eight worldwide levels: Serbia-only
    data makes `country` constant and `region` near-constant.

Requires transformers<5.0 — see geoclip_adaptation/model.py.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from geoclip.model.image_encoder import ImageEncoder
from geoclip.model.location_encoder import LocationEncoder
from peft import LoraConfig, get_peft_model

_PROJ_NAMES = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "out": "out_proj"}


def _vision_attn_pattern(targets):
    return rf".*vision_model.*({'|'.join(_PROJ_NAMES[t] for t in targets)})$"


class G3GeoAlignment(nn.Module):
    """Image <-> GPS contrastive alignment, InfoNCE with a GPS queue."""

    def __init__(
        self,
        image_adaptation="frozen_mlp",
        location_pretrained=True,
        lora_targets=("q", "k", "v", "out"),
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        queue_size=4096,
        use_text=True,
    ):
        super().__init__()
        assert image_adaptation in ("frozen_mlp", "lora")
        self.image_adaptation = image_adaptation
        self.use_text = use_text

        # ImageEncoder() freezes CLIP's parameters in its own __init__ and
        # leaves its 768->768->512 MLP trainable, so "frozen_mlp" needs no
        # further action here.
        self.image_encoder = ImageEncoder()
        if image_adaptation == "lora":
            cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=_vision_attn_pattern(lora_targets),
                lora_dropout=lora_dropout, bias="none",
            )
            self.image_encoder.CLIP = get_peft_model(self.image_encoder.CLIP, cfg)
            self.image_encoder.CLIP.visual_projection.weight.requires_grad_(True)

        # from_pretrained=True starts from GeoCLIP's globally-trained location
        # encoder; False trains the random-Fourier-feature MLP from scratch.
        #
        # NOTE this is a deliberate deviation from G3. G3 explicitly rejects
        # GeoCLIP's equal-earth projection in favour of Mercator ("EEP
        # primarily focuses on area accuracy while overlooking angular
        # distortions"), so a faithful G3 location encoder would be Mercator
        # based and randomly initialised. We use GeoCLIP's EEP-based one
        # pretrained instead, for two measured reasons:
        #
        #   1. Its weights come from 4.7M worldwide coordinates. Training an
        #      equivalent encoder from scratch on 16,294 Serbian points would
        #      almost certainly do worse.
        #   2. G3's objection to EEP is a global-scale argument. Over Serbia's
        #      actual extent (lat 42.26-46.13, lon 18.92-22.85) the two
        #      projections differ by an affine map plus a residual under 0.9%
        #      of each axis range, and an affine difference is exactly what
        #      the RFF+MLP stack in front of it can absorb.
        self.location_encoder = LocationEncoder(from_pretrained=location_pretrained)

        # Text branch: CLIP's text tower already exists inside the CLIPModel
        # that ImageEncoder loads (unused by GeoCLIP), so no extra model is
        # downloaded. It stays frozen; only this projection trains, mirroring
        # the image side's 768->768->512 MLP.
        if use_text:
            self.text_projection = nn.Sequential(
                nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512),
            )
            self.register_buffer("text_feature_cache", torch.zeros(0, 768), persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # InfoNCE with batch_size=32 would see only 31 negatives per anchor,
        # far too few to learn a discriminative space. GeoCLIP's fix, which
        # G3 inherits: a rolling queue of past GPS coordinates appended as
        # extra negatives, giving queue_size + batch_size - 1 per anchor.
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, queue_size))
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        b = gps.size(0)
        assert self.queue_size % b == 0, (
            f"queue_size ({self.queue_size}) must be divisible by batch size ({b})"
        )
        ptr = int(self.gps_queue_ptr)
        self.gps_queue[:, ptr:ptr + b] = gps.t()
        self.gps_queue_ptr[0] = (ptr + b) % self.queue_size

    @torch.no_grad()
    def precompute_text_features(self, vocab, device, batch_size=64):
        """Run the frozen text tower once over every unique description.

        There are only a few hundred distinct place strings across the split,
        versus 16k images, so caching here removes the text tower from the
        training loop almost entirely — only the small projection stays live.
        Valid precisely because the tower is frozen: its output per string
        never changes during training.
        """
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        clip = self.image_encoder.CLIP
        # get_peft_model wraps CLIPModel; the text tower is untouched by the
        # vision-scoped LoRA, but the wrapper still forwards attribute access.
        feats = []
        for i in range(0, len(vocab), batch_size):
            batch = vocab[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=77, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            feats.append(clip.get_text_features(**enc).float().cpu())
        self.text_feature_cache = torch.cat(feats, dim=0).to(device)
        return self.text_feature_cache.shape

    def encode_image(self, pixel_values):
        return F.normalize(self.image_encoder(pixel_values), dim=1)

    def encode_location(self, gps):
        return F.normalize(self.location_encoder(gps), dim=1)

    def encode_text(self, text_idx):
        assert self.text_feature_cache.numel() > 0, (
            "call precompute_text_features(vocab, device) before training"
        )
        return F.normalize(self.text_projection(self.text_feature_cache[text_idx]), dim=1)

    def forward(self, pixel_values, gps, text_idx=None):
        """Returns (gps_logits, text_logits).

        gps_logits  [B, B + queue_size] — correct GPS for row i is column i,
                    remaining columns are queue negatives.
        text_logits [B, B] or None — in-batch only; duplicate descriptions
                    within a batch are masked by the caller, since several
                    images can share a city.
        """
        img_f = self.encode_image(pixel_values)

        gps_all = torch.cat([gps, self.gps_queue.t().detach().to(gps.device)], dim=0)
        scale = self.logit_scale.exp()
        gps_logits = scale * (img_f @ self.encode_location(gps_all).t())

        text_logits = None
        if self.use_text and text_idx is not None:
            text_logits = scale * (img_f @ self.encode_text(text_idx).t())
        return gps_logits, text_logits

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
