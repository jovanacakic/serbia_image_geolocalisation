"""
g3_adaptation/train_alignment.py — trains G3's geo-alignment component
(stage 1) on Serbia, then evaluates it the way G3 actually predicts: by
geo-verification (stage 3) over a candidate gallery.

Inference contrast with the other two models in this project:
  GeoToken           retrieves neighbours from the train gallery as *context*
                     for an autoregressive S2-token decoder.
  GeoCLIP-adapted    discards retrieval entirely, regressing (lat, lon).
  G3 (this)          keeps retrieval as the prediction mechanism, but scores
                     candidates in a *learned* joint image/GPS space rather
                     than by raw CLIP image-image similarity.

Why retrieval is worth keeping here, unlike for GeoCLIP: the ceiling depends
entirely on gallery density. GeoCLIP's shipped 100K-point global gallery has
only 70 points inside Serbia, capping any retrieval method at ~20.4km median.
Retrieving over the 16,294 Serbia train coordinates instead has an oracle
ceiling of ~0.35km median (72% within 1km) — so here the gallery is not the
bottleneck, the image/GPS alignment is.

Stage 1 is tri-modal by default (image / GPS / text), matching G3's own
formulation; --no_text drops the text terms, reducing it to image<->GPS,
which is exactly GeoCLIP's formulation and serves as the ablation isolating
what text contributes. Text is training-time only — geo-verification at
inference is image<->GPS, so no text is needed to predict.

Usage (Colab, after %cd into the project root):
  # faithful G3, tri-modal (run this first)
  !python g3_adaptation/train_alignment.py --run_name g3_text \
      --image_adaptation frozen_mlp --lr 1e-4 --epochs 20
  !python g3_adaptation/train_alignment.py --run_name g3_text \
      --image_adaptation frozen_mlp --lr 1e-4 --epochs 35 \
      --resume g3_adaptation/checkpoints/g3_text/last_model.pt

  # ablation: no text modality
  !python g3_adaptation/train_alignment.py --run_name g3_no_text \
      --image_adaptation frozen_mlp --lr 1e-4 --epochs 20 --no_text

Any --image_adaptation lora run needs `!pip uninstall -y -q torchao` first on
Colab — see geoclip_adaptation/train_lora.py's docstring. --image_adaptation
frozen_mlp does not touch PEFT and needs no such fix.
"""
import os
import sys
import csv
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from g3_adaptation.model import G3GeoAlignment
from g3_adaptation.dataset import SerbiaG3Dataset
from geoclip_adaptation.dataset import SerbiaGeoCLIPDataset
from geoclip_adaptation.train_lora import (
    sync_images_to_local, sanity_check_images,
    DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR,
)
from utils.eval_utils import haversine_distance, format_metrics
from utils.checkpoint_utils import trainable_state_dict, load_any_state_dict

CSV_PATH = os.path.join(BASE_DIR, "data", "osv5m_serbia_metadata.csv")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "processed", "splits")


def load_gallery_coords():
    """G3's geo-verification candidate pool: the train split's coordinates.
    Train-only — using val/test coordinates as candidates would leak the
    answer set into evaluation."""
    df_all = pd.read_csv(CSV_PATH)
    idx = torch.load(os.path.join(SPLITS_DIR, "train_indices.pt"), weights_only=True)
    df = df_all.iloc[idx.tolist()]
    return torch.tensor(df[["latitude", "longitude"]].values, dtype=torch.float32)


def info_nce_gps(logits):
    """Symmetric image<->GPS InfoNCE. Columns 0..B-1 are the in-batch GPS (so
    image i matches GPS i); the remaining columns are queue negatives, which
    have no image counterpart and so appear only in the image->GPS direction."""
    b = logits.size(0)
    targets = torch.arange(b, device=logits.device)
    return F.cross_entropy(logits, targets), F.cross_entropy(logits[:, :b].t(), targets)


def info_nce_text(logits, text_idx):
    """Symmetric image<->text InfoNCE with duplicate masking.

    Unlike GPS (continuous, effectively unique per image), place descriptions
    repeat: with 313 cities and batch 32, several images routinely share a
    description. Those are genuine positives, so leaving them in the
    denominator as "negatives" would train the model to push apart images
    that belong together. Mask them out; the diagonal stays as the target.
    """
    b = logits.size(0)
    targets = torch.arange(b, device=logits.device)
    dup = text_idx.unsqueeze(1) == text_idx.unsqueeze(0)
    off_diag_dup = dup & ~torch.eye(b, dtype=torch.bool, device=logits.device)
    masked = logits.masked_fill(off_diag_dup, float("-inf"))
    return (F.cross_entropy(masked, targets),
            F.cross_entropy(masked.t(), targets))


def g3_loss(gps_logits, text_logits, text_idx):
    """L = (L_img,txt + L_img,gps + L_txt,img + L_gps,img) / 4.
    With --no_text only the two GPS terms exist, averaged over 2.

    NOTE the constant is ours, not G3's. G3's Eq. 2 divides the same four
    terms by 2, as does GeoToken's restatement of the objective. Dividing by
    4 scales the loss, and therefore every gradient, by one half, which is
    equivalent to halving the learning rate; it does not change what is being
    optimised. Reported results were trained this way, so the constant is
    left as-is rather than changed after the fact."""
    l_i2g, l_g2i = info_nce_gps(gps_logits)
    if text_logits is None:
        return 0.5 * (l_i2g + l_g2i)
    l_i2t, l_t2i = info_nce_text(text_logits, text_idx)
    return 0.25 * (l_i2t + l_i2g + l_t2i + l_g2i)


@torch.no_grad()
def evaluate(model, loader, gallery_coords, device, use_amp):
    """Geo-verification: pick the gallery coordinate whose GPS embedding is
    most similar to the query image's embedding."""
    model.eval()
    gallery_dev = gallery_coords.to(device)
    with autocast("cuda", enabled=use_amp):
        gallery_emb = model.encode_location(gallery_dev)
    gallery_emb = gallery_emb.float()

    preds, trues = [], []
    for pixel_values, coords in loader:
        with autocast("cuda", enabled=use_amp):
            img_emb = model.encode_image(pixel_values.to(device))
        sims = img_emb.float() @ gallery_emb.t()
        preds.append(gallery_dev[sims.argmax(dim=1)].cpu())
        trues.append(coords)
    model.train()

    pred_coords = torch.cat(preds, dim=0)
    true_coords = torch.cat(trues, dim=0)
    distances = haversine_distance(pred_coords, true_coords)
    metrics = {f"acc@{t}km": (distances <= t).float().mean().item() * 100
               for t in (1, 25, 200, 750, 2500)}
    metrics["median_km"] = distances.median().item()
    metrics["mean_km"] = distances.mean().item()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--image_adaptation", choices=["frozen_mlp", "lora"], default="frozen_mlp")
    parser.add_argument("--lora_targets", type=str, default="q,k,v,out")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--location_pretrained", type=int, default=1,
                        help="1: start from GeoCLIP's pretrained location encoder; 0: from scratch.")
    parser.add_argument("--queue_size", type=int, default=4096)
    parser.add_argument("--no_text", action="store_true",
                        help="Ablation: drop G3's text modality, reducing stage 1 to "
                             "image<->GPS alignment (i.e. GeoCLIP's formulation).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--scheduler", choices=["cosine", "warmup_cosine"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    print("=" * 65)
    print(f"TRAINING G3 Geo-Alignment — run: {args.run_name}")
    print("=" * 65)
    print("Hyperparameters:")
    for k, v in vars(args).items():
        if k != "run_name":
            print(f"  {k:<22}: {v}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print(f"Training on device: {device}")

    checkpoint_dir = os.path.join(BASE_DIR, "g3_adaptation", "checkpoints", args.run_name)
    tb_log_dir = os.path.join(BASE_DIR, "g3_adaptation", "tensorboard_logs", args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    model = G3GeoAlignment(
        image_adaptation=args.image_adaptation,
        location_pretrained=bool(args.location_pretrained),
        lora_targets=tuple(t.strip() for t in args.lora_targets.split(",")),
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        queue_size=args.queue_size,
        use_text=not args.no_text,
    ).to(device)
    print(f"Trainable params: {model.trainable_parameter_count():,}")

    raw_images_dir = sync_images_to_local(DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR)
    train_ds = SerbiaG3Dataset(
        CSV_PATH, raw_images_dir, os.path.join(SPLITS_DIR, "train_indices.pt"), train=True)
    val_ds = SerbiaGeoCLIPDataset(
        CSV_PATH, raw_images_dir, os.path.join(SPLITS_DIR, "val_indices.pt"), train=False)
    sanity_check_images(train_ds.df, raw_images_dir)
    # drop_last is required, not cosmetic: the GPS queue asserts
    # queue_size % batch_size == 0, which a short final batch would violate.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    gallery_coords = load_gallery_coords()
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Gallery: {gallery_coords.size(0)} coords")
    if not args.no_text:
        shape = model.precompute_text_features(train_ds.vocab, device)
        print(f"Text modality: {len(train_ds.vocab)} unique descriptions, cached {tuple(shape)}")
        print(f'  e.g. "{train_ds.vocab[0]}"')
    else:
        print("Text modality: DISABLED (--no_text ablation)")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch, best_val_median_km = 1, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        load_any_state_dict(model, ckpt)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_median_km = ckpt["best_val_median_km"]
        print(f"Resuming at epoch {start_epoch}, best-so-far: {best_val_median_km:.1f}km")
        if start_epoch > args.epochs:
            print(f"[ERROR] --epochs ({args.epochs}) must exceed checkpoint epoch ({ckpt['epoch']}).")
            sys.exit(1)

    # Scoped to the remaining epochs — see geoclip_adaptation/train_lora.py.
    remaining = args.epochs - start_epoch + 1
    total_steps = len(train_loader) * remaining
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.eta_min)
    else:
        def lr_lambda(step):
            if step < args.warmup_steps:
                return (step + 1) / max(1, args.warmup_steps)
            prog = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
            floor = args.eta_min / args.lr
            return floor + (1 - floor) * 0.5 * (1 + torch.cos(torch.tensor(prog * 3.14159265)).item())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    csv_path = os.path.join(checkpoint_dir, "training_log.csv")
    csv_fields = ["epoch", "train_loss", "val_median_km", "val_mean_km",
                  "acc@1km", "acc@25km", "acc@200km", "acc@750km", "acc@2500km", "lr"]
    if not args.resume:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_fields)

    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled (no GPU)'}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}")
        for pixel_values, coords, text_idx in pbar:
            pixel_values, coords = pixel_values.to(device), coords.to(device)
            text_idx = text_idx.to(device)
            with autocast("cuda", enabled=use_amp):
                gps_logits, text_logits = model(pixel_values, coords, text_idx)
            loss = g3_loss(gps_logits.float(),
                           None if text_logits is None else text_logits.float(),
                           text_idx)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.dequeue_and_enqueue(coords)

            # Keep the temperature in CLIP's usual range; an unclamped
            # logit_scale can run away and collapse the contrastive signal.
            with torch.no_grad():
                model.logit_scale.clamp_(0, 4.6052)  # ln(100)

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", temp=f"{model.logit_scale.exp().item():.1f}")

        train_loss = running_loss / n_batches
        current_lr = optimizer.param_groups[0]["lr"]
        val_metrics = evaluate(model, val_loader, gallery_coords, device, use_amp)

        print(f" Epoch {epoch:02d} [TRAIN] loss={train_loss:.4f} | "
              f"[VAL] {format_metrics(val_metrics)}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Haversine_km/val_median", val_metrics["median_km"], epoch)
        writer.add_scalar("LR", current_lr, epoch)
        for t in (1, 25, 200, 750, 2500):
            writer.add_scalar(f"Accuracy/{t}km", val_metrics[f"acc@{t}km"], epoch)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_metrics["median_km"], val_metrics["mean_km"],
                val_metrics["acc@1km"], val_metrics["acc@25km"], val_metrics["acc@200km"],
                val_metrics["acc@750km"], val_metrics["acc@2500km"], current_lr,
            ])

        if val_metrics["median_km"] < best_val_median_km:
            best_val_median_km = val_metrics["median_km"]
            torch.save(trainable_state_dict(model),
                       os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"  [SAVED] New best val median: {best_val_median_km:.1f}km")

        torch.save({
            "epoch": epoch,
            "model_state_dict": trainable_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_median_km": best_val_median_km,
        }, os.path.join(checkpoint_dir, "last_model.pt"))

    writer.close()
    print("=" * 65)
    print(f"[TRAINING COMPLETE] Best val median error: {best_val_median_km:.1f}km")
    print(f"Checkpoint: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"Training log: {csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
