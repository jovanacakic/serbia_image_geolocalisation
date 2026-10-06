"""
evaluate_test.py — the single, shared evaluation harness for every model in
this project, so all reported numbers are directly comparable.

One pipeline, one metric set, one split, applied identically to baselines
and to every adapted model. Comparing a retrieval model against
a regression model against an autoregressive decoder is only meaningful if
nothing differs except the model itself.

  --model geotoken   frozen-CLIP embeddings + retrieved context -> autoregressive
                     S2-token decode -> coordinates
  --model geoclip    LoRA/full-unfreeze CLIP -> regression head -> coordinates
                     (supports --tta, 6-view test-time averaging)
  --model g3         contrastive image/GPS alignment -> geo-verification
                     (argmax similarity over the train-coordinate gallery)
  --model centroid   always predict the training centroid (reference floor)

IMPORTANT — the test split is evaluated ONCE, after every model and
hyperparameter decision is locked in. Run with `--split val` first to confirm
a command reproduces that model's known validation number; only then rerun it
with `--split test`. The script prints a warning whenever --split test is used.

Usage (Colab, after %cd into the project root):
  # sanity-check on val first
  !python evaluate_test.py --model geoclip --split val \
      --checkpoint geoclip_adaptation/checkpoints/v1_full_unfreeze/best_model.pt \
      --adaptation full_unfreeze

  # then, once, for real
  !python evaluate_test.py --model geoclip --split test \
      --checkpoint geoclip_adaptation/checkpoints/v1_full_unfreeze/best_model.pt \
      --adaptation full_unfreeze --tta
"""
import os
import sys
import csv
import time
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from utils.eval_utils import haversine_distance, format_metrics
from utils.checkpoint_utils import load_any_state_dict

CSV_PATH = os.path.join(BASE_DIR, "data", "osv5m_serbia_metadata.csv")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "processed", "splits")

CLIP_MEAN = [0.481, 0.458, 0.408]
CLIP_STD = [0.269, 0.261, 0.276]


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def compute_metrics(pred_coords, true_coords):
    d = haversine_distance(pred_coords, true_coords)
    m = {f"acc@{t}km": (d <= t).float().mean().item() * 100 for t in (1, 25, 200, 750, 2500)}
    m["median_km"] = d.median().item()
    m["mean_km"] = d.mean().item()
    return m, d


def load_split_coords(split):
    df = pd.read_csv(CSV_PATH)
    idx = torch.load(os.path.join(SPLITS_DIR, f"{split}_indices.pt"), weights_only=True)
    sub = df.iloc[idx.tolist()].reset_index(drop=True)
    return torch.tensor(sub[["latitude", "longitude"]].values, dtype=torch.float32)


# --- Test-time augmentation -------------------------------------------------
# Six deterministic views; predictions are averaged. Deterministic matters:
# TTA must be reproducible, so these are fixed transforms, not the random
# ones used for training augmentation.
def tta_views():
    norm = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    to_t = transforms.ToTensor()
    return [
        ("resize", transforms.Compose([transforms.Resize((224, 224)), to_t, norm])),
        ("hflip", transforms.Compose([
            transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0), to_t, norm])),
        ("center_crop", transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), to_t, norm])),
        ("center_crop_flip", transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0), to_t, norm])),
        ("bright_contrast", transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=(1.2, 1.2), contrast=(1.2, 1.2)), to_t, norm])),
        ("rot10", transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation((10, 10)), to_t, norm])),
    ]


class ImageSplitDataset(torch.utils.data.Dataset):
    """Returns one tensor per TTA view (a single 'resize' view when TTA is off)."""

    def __init__(self, split, img_dir, use_tta):
        df = pd.read_csv(CSV_PATH)
        idx = torch.load(os.path.join(SPLITS_DIR, f"{split}_indices.pt"), weights_only=True)
        self.df = df.iloc[idx.tolist()].reset_index(drop=True)
        self.img_dir = img_dir
        views = tta_views()
        self.views = views if use_tta else views[:1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.img_dir, f"{row['id']}.jpg")
        img = Image.open(path).convert("RGB") if os.path.exists(path) \
            else Image.new("RGB", (224, 224), (0, 0, 0))
        stack = torch.stack([t(img) for _, t in self.views])  # [V, 3, 224, 224]
        coords = torch.tensor([row["latitude"], row["longitude"]], dtype=torch.float32)
        return stack, coords


def get_image_loader(split, use_tta, batch_size, num_workers):
    from geoclip_adaptation.train_lora import (
        sync_images_to_local, DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR)
    img_dir = sync_images_to_local(DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR)
    ds = ImageSplitDataset(split, img_dir, use_tta)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# --- Per-model prediction ---------------------------------------------------
@torch.no_grad()
def predict_geoclip(args, device):
    from geoclip_adaptation.model import GeoCLIPRegressor

    model = GeoCLIPRegressor(
        adaptation=args.adaptation,
        lora_targets=tuple(t.strip() for t in args.lora_targets.split(",")),
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
    ).to(device)
    load_any_state_dict(model, torch.load(resolve(args.checkpoint),
                                          map_location=device, weights_only=True))
    model.eval()

    use_amp = (device == "cuda")
    loader = get_image_loader(args.split, args.tta, args.batch_size, args.num_workers)
    preds, trues = [], []
    for stack, coords in tqdm(loader, desc="  geoclip"):
        b, v = stack.shape[:2]
        flat = stack.view(b * v, *stack.shape[2:]).to(device)
        # autocast to match the regime the model was trained and validated
        # in — see predict_g3 for why this is not merely cosmetic.
        with autocast("cuda", enabled=use_amp):
            out = model(flat)
        out = out.float().view(b, v, 2)      # [B, V, 2]
        preds.append(out.mean(dim=1).cpu())  # average coordinates across views
        trues.append(coords)
    return torch.cat(preds), torch.cat(trues)


@torch.no_grad()
def predict_g3(args, device):
    from g3_adaptation.model import G3GeoAlignment

    model = G3GeoAlignment(
        image_adaptation=args.image_adaptation,
        lora_targets=tuple(t.strip() for t in args.lora_targets.split(",")),
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    ).to(device)
    load_any_state_dict(model, torch.load(resolve(args.checkpoint),
                                          map_location=device, weights_only=True))
    model.eval()

    use_amp = (device == "cuda")

    # autocast is REQUIRED here, not cosmetic. The location encoder uses
    # Random Fourier Features with sigma up to 2**8; sin/cos of large
    # arguments is exactly where fp16 loses precision, so fp32 and fp16
    # produce materially different location embeddings (measured: mean
    # cosine 0.939, and 77% of retrieval argmax decisions flip). The model
    # was trained and checkpoint-selected under AMP, so inference must use
    # the same numerical regime or its predictions do not match the
    # validation numbers the checkpoint was chosen by.
    #
    # Candidate pool is the TRAIN coordinates only — val/test coordinates as
    # candidates would hand the model its own answer set.
    gallery = load_split_coords("train").to(device)
    with autocast("cuda", enabled=use_amp):
        gallery_emb = model.encode_location(gallery)
    gallery_emb = gallery_emb.float()

    loader = get_image_loader(args.split, args.tta, args.batch_size, args.num_workers)
    preds, trues = [], []
    for stack, coords in tqdm(loader, desc="  g3"):
        b, v = stack.shape[:2]
        flat = stack.view(b * v, *stack.shape[2:]).to(device)
        with autocast("cuda", enabled=use_amp):
            emb = model.encode_image(flat)
        emb = emb.float().view(b, v, -1).mean(dim=1)
        # Averaging unit vectors denormalizes them; renormalize so the
        # similarity below stays a true cosine.
        emb = F.normalize(emb, dim=1)
        preds.append(gallery[(emb @ gallery_emb.t()).argmax(dim=1)].cpu())
        trues.append(coords)
    return torch.cat(preds), torch.cat(trues)


@torch.no_grad()
def predict_geotoken(args, device):
    from geotoken.transformer_model.transformer_architecture import GeoTokenTransformer
    from geotoken.context_retrieval.search_neighbors import GallerySearcher
    from utils.s2_geometry_helpers import s2_tokens_to_coords

    feat_path = args.features_path or f"geotoken/context_retrieval/{args.split}_features.pt"
    features = torch.load(resolve(feat_path), weights_only=True)
    searcher = GallerySearcher(gallery_path=resolve(args.gallery_path), device=device)

    model = GeoTokenTransformer(
        clip_dim=args.clip_dim, d_model=512, nhead=8,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
    ).to(device)
    ckpt = torch.load(resolve(args.checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    embeddings, true_coords = features["embeddings"], features["coords"]
    preds = []
    for start in tqdm(range(0, embeddings.size(0), args.batch_size), desc="  geotoken"):
        emb = embeddings[start:start + args.batch_size].to(device)
        n_s2, n_emb, _ = searcher.find_neighbors(emb, top_k=args.top_k, exclude_self=False)
        for seq in model.generate_greedy(emb, n_emb, n_s2).cpu().tolist():
            preds.append(list(s2_tokens_to_coords(seq)))
    return torch.tensor(preds, dtype=torch.float32), true_coords


def predict_centroid(args, device):
    centroid = load_split_coords("train").mean(dim=0)
    true = load_split_coords(args.split)
    return centroid.unsqueeze(0).expand(true.size(0), 2), true


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["geotoken", "geoclip", "g3", "centroid"], required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--label", type=str, default=None,
                   help="Row name in the results CSV. Defaults to model+checkpoint dir.")
    p.add_argument("--tta", action="store_true",
                   help="6-view test-time augmentation. geoclip/g3 only.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    # geoclip
    p.add_argument("--adaptation", choices=["lora", "full_unfreeze"], default="lora")
    p.add_argument("--unfreeze_last_n_layers", type=int, default=2)
    # g3
    p.add_argument("--image_adaptation", choices=["frozen_mlp", "lora"], default="frozen_mlp")
    # shared lora
    p.add_argument("--lora_targets", type=str, default="q,k,v,out")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    # geotoken
    p.add_argument("--gallery_path", type=str,
                   default="geotoken/context_retrieval/gallery_database_train_val.pt")
    p.add_argument("--top_k", type=int, default=15)
    p.add_argument("--clip_dim", type=int, default=768)
    p.add_argument("--features_path", type=str, default=None,
                   help="Override the geotoken feature file for this split.")
    p.add_argument("--num_encoder_layers", type=int, default=6)
    p.add_argument("--num_decoder_layers", type=int, default=6)
    p.add_argument("--dim_feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--output_csv", type=str, default="final_results.csv")
    args = p.parse_args()

    if args.model != "centroid" and not args.checkpoint:
        p.error(f"--checkpoint is required for --model {args.model}")
    if args.tta and args.model in ("geotoken", "centroid"):
        p.error(f"--tta is not applicable to --model {args.model} "
                f"(no continuous output / no image input to augment)")

    print("=" * 65)
    print(f"EVALUATION — model: {args.model} | split: {args.split}"
          f"{' | TTA' if args.tta else ''}")
    print("=" * 65)
    if args.split == "test":
        print("*** TEST SPLIT — this is the one-time final measurement. ***")
        print("*** Confirm the same command reproduces the expected val  ***")
        print("*** number first if you have not already done so.         ***")
        print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if args.checkpoint:
        print(f"Checkpoint: {resolve(args.checkpoint)}")

    predictor = {
        "geoclip": predict_geoclip, "g3": predict_g3,
        "geotoken": predict_geotoken, "centroid": predict_centroid,
    }[args.model]
    pred_coords, true_coords = predictor(args, device)

    metrics, distances = compute_metrics(pred_coords, true_coords)
    label = args.label or (
        f"{args.model}"
        + (f"/{os.path.basename(os.path.dirname(resolve(args.checkpoint)))}" if args.checkpoint else "")
        + (" +TTA" if args.tta else "")
    )

    print("\n" + "=" * 65)
    print(f"RESULTS — {label} ({args.split}, {true_coords.size(0)} images)")
    print("=" * 65)
    print(format_metrics(metrics))
    print(f"mean={metrics['mean_km']:.1f}km")
    for q in (0.9, 0.95):
        print(f"p{int(q * 100)}={torch.quantile(distances, q):.1f}km")
    print("=" * 65)

    out = resolve(args.output_csv)
    fields = (["timestamp", "split", "model", "label", "checkpoint", "tta",
               "median_km", "mean_km"] + [f"acc@{t}km" for t in (1, 25, 200, 750, 2500)])
    write_header = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(fields)
        w.writerow(
            [time.strftime("%Y-%m-%d %H:%M:%S"), args.split, args.model, label,
             args.checkpoint or "", int(args.tta),
             metrics["median_km"], metrics["mean_km"]]
            + [metrics[f"acc@{t}km"] for t in (1, 25, 200, 750, 2500)])
    print(f"Appended to: {out}")


if __name__ == "__main__":
    main()
