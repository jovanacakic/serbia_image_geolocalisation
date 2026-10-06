"""
baselines/g3_zeroshot.py — the official pretrained G3 (Jia et al., NeurIPS
2024), evaluated zero-shot on Serbia against the SAME candidate gallery our
own g3_adaptation model uses.

Why this baseline is worth more than the GeoCLIP zero-shot rows: it varies
exactly one thing. Same architecture, same 16,294-coordinate Serbia gallery,
same metrics — the only difference is worldwide-pretrained alignment weights
versus ours trained on Serbia. That isolates what the domain adaptation
actually bought, where "GeoCLIP zero-shot vs our GeoCLIP" confounds the
alignment, the gallery, and the output head all at once.

Their released g3.index (38 GB) is NOT needed: that indexes their MP16-Pro
gallery, and we deliberately retrieve over the Serbia gallery instead so the
candidate pool is held constant.

Setup on Colab (downloads to /content, not Drive — 1.77 GB, re-fetchable):
    !pip install -q pyproj huggingface_hub
    !git clone https://github.com/Applied-Machine-Learning-Lab/G3.git /content/G3_official
    !huggingface-cli download Jia-py/G3-checkpoint g3.pth --local-dir /content/g3_ckpt

Usage:
    !python baselines/g3_zeroshot.py \
        --g3_repo /content/G3_official \
        --checkpoint /content/g3_ckpt/g3.pth

GPU only: their CustomLocationEncoder hardcodes .to('cuda') internally, so
there is no CPU path to fall back on.
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from utils.eval_utils import haversine_distance, format_metrics

CSV_PATH = os.path.join(BASE_DIR, "data", "osv5m_serbia_metadata.csv")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "processed", "splits")


class PILSplitDataset(Dataset):
    """Returns raw PIL images: their CLIPImageProcessor does its own resize
    and centre-crop, so we must not pre-transform with our own pipeline."""

    def __init__(self, split, img_dir):
        df = pd.read_csv(CSV_PATH)
        idx = torch.load(os.path.join(SPLITS_DIR, f"{split}_indices.pt"), weights_only=True)
        self.df = df.iloc[idx.tolist()].reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.img_dir, f"{row['id']}.jpg")
        img = (Image.open(path).convert("RGB") if os.path.exists(path)
               else Image.new("RGB", (224, 224), (0, 0, 0)))
        return img, torch.tensor([row["latitude"], row["longitude"]], dtype=torch.float32)


def collate(batch):
    return [b[0] for b in batch], torch.stack([b[1] for b in batch])


def load_coords(split):
    df = pd.read_csv(CSV_PATH)
    idx = torch.load(os.path.join(SPLITS_DIR, f"{split}_indices.pt"), weights_only=True)
    sub = df.iloc[idx.tolist()]
    return torch.tensor(sub[["latitude", "longitude"]].values, dtype=torch.float32)


@torch.no_grad()
def encode_locations(model, coords, device, chunk=2048):
    """location_encoder -> location_projection_else -> normalize.
    Chunked because their encoder round-trips through numpy/pyproj on CPU."""
    out = []
    for s in range(0, coords.size(0), chunk):
        emb = model.location_encoder(coords[s:s + chunk].to(device))
        out.append(F.normalize(model.location_projection_else(emb), dim=-1).float())
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_images(model, loader, device):
    """vision_model[1] (pooler) -> vision_projection -> vision_projection_else_2
    -> normalize. Mirrors the 'phase _2' branch of their forward()."""
    embs, trues = [], []
    for images, coords in tqdm(loader, desc="  encoding images"):
        px = model.vision_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        v = model.vision_model(px)[1]
        e = model.vision_projection_else_2(model.vision_projection(v))
        embs.append(F.normalize(e, dim=-1).float())
        trues.append(coords)
    return torch.cat(embs, dim=0), torch.cat(trues, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--g3_repo", type=str, default="/content/G3_official")
    p.add_argument("--checkpoint", type=str, default="/content/g3_ckpt/g3.pth")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--output_csv", type=str, default="baselines/baseline_results.csv")
    args = p.parse_args()

    print("=" * 65)
    print(f"BASELINE: pretrained G3 (zero-shot) — split: {args.split}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[FATAL] Their CustomLocationEncoder hardcodes .to('cuda'); "
              "this baseline cannot run on CPU.")
        sys.exit(1)

    assert os.path.isdir(args.g3_repo), f"G3 repo not found: {args.g3_repo}"
    assert os.path.exists(args.checkpoint), f"checkpoint not found: {args.checkpoint}"

    # Their utils/G3.py uses a relative import (from .rff.layers import ...),
    # so the repo ROOT goes on sys.path and we import it as a package.
    sys.path.insert(0, args.g3_repo)
    from utils.G3 import G3

    print("Loading pretrained G3 ...")
    model = G3(device).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [warn] missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        for k in list(missing)[:5]:
            print(f"    missing: {k}")
        for k in list(unexpected)[:5]:
            print(f"    unexpected: {k}")
    else:
        print("  state_dict loaded cleanly (strict match)")
    model.eval()

    from geoclip_adaptation.train_lora import (
        sync_images_to_local, DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR)
    img_dir = sync_images_to_local(DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR)

    ds = PILSplitDataset(args.split, img_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    gallery = load_coords("train")
    print(f"{args.split}: {len(ds)} images | gallery: {gallery.size(0)} train coords")

    gallery_emb = encode_locations(model, gallery, device)
    img_emb, true_coords = encode_images(model, loader, device)

    # Geo-verification: highest cosine similarity between the image and each
    # candidate coordinate, in their 768-d joint space.
    preds = []
    for s in range(0, img_emb.size(0), 256):
        sims = img_emb[s:s + 256] @ gallery_emb.t()
        preds.append(gallery[sims.argmax(dim=1).cpu()])
    pred_coords = torch.cat(preds, dim=0)

    d = haversine_distance(pred_coords, true_coords)
    metrics = {f"acc@{t}km": (d <= t).float().mean().item() * 100
               for t in (1, 25, 200, 750, 2500)}
    metrics["median_km"] = d.median().item()
    metrics["mean_km"] = d.mean().item()

    print("\n" + "=" * 65)
    print(f"RESULTS — pretrained G3 zero-shot ({args.split}, {len(ds)} images)")
    print("=" * 65)
    print(format_metrics(metrics))
    print(f"mean={metrics['mean_km']:.1f}km")
    print("=" * 65)

    out = args.output_csv if os.path.isabs(args.output_csv) \
        else os.path.join(BASE_DIR, args.output_csv)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = (["timestamp", "split", "baseline", "median_km", "mean_km"]
              + [f"acc@{t}km" for t in (1, 25, 200, 750, 2500)])
    write_header = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(fields)
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), args.split,
                    "G3 pretrained (zero-shot, Serbia gallery)",
                    metrics["median_km"], metrics["mean_km"]]
                   + [metrics[f"acc@{t}km"] for t in (1, 25, 200, 750, 2500)])
    print(f"Appended to: {out}")


if __name__ == "__main__":
    main()
