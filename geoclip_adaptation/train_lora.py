"""
geoclip_adaptation/train_lora.py — fine-tunes GeoCLIP for direct GPS
regression on Serbia (~16k train images).

Why regression instead of GeoCLIP's native retrieval: GeoCLIP ships a
~100K-point global GPS gallery, of which only 70 points fall inside Serbia.
Retrieval accuracy is therefore capped by that gallery's local density
rather than by the image encoder's quality (measured ceiling: ~20.4km median),
whereas a regression head can output any continuous point inside the country.

This script supports the full adaptation ladder (full-unfreeze, LoRA{q,v}
r=8, LoRA{q,k,v,out} r=16, refined LR) as separate --run_name runs, so each
design decision can be measured on Serbia's own data.

Usage (Colab, after %cd into the project root):
  !python geoclip_adaptation/train_lora.py --run_name v1_full_unfreeze \
      --adaptation full_unfreeze --unfreeze_last_n_layers 2 --lr 1e-4 --epochs 15

  !python geoclip_adaptation/train_lora.py --run_name v2_lora_qv_r8 \
      --adaptation lora --lora_targets q,v --lora_r 8 --lr 1e-4 --epochs 30

  !python geoclip_adaptation/train_lora.py --run_name v3_lora_qkvout_r16 \
      --adaptation lora --lora_targets q,k,v,out --lora_r 16 --lr 1e-4 --epochs 30

  !python geoclip_adaptation/train_lora.py --run_name v4_refined_lr \
      --adaptation lora --lora_targets q,k,v,out --lora_r 16 \
      --lr 5e-5 --scheduler warmup_cosine --warmup_steps 50 --epochs 30

Any --adaptation lora run needs a one-time environment fix on Colab first:
Colab preinstalls torchao==0.10.0, and peft's get_peft_model() unconditionally
checks torchao's version while dispatching LoRA target modules, raising
ImportError on anything below 0.16.0 — even though this project never uses
torchao at all. --adaptation full_unfreeze never hits this code path, which
is why v1 ran fine before this surfaced. Fix, once per Colab session:
    !pip uninstall -y -q torchao
"""
import os
import sys
import csv
import shutil
import subprocess
import argparse

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from geoclip_adaptation.model import GeoCLIPRegressor
from geoclip_adaptation.dataset import SerbiaGeoCLIPDataset
from utils.eval_utils import haversine_distance, format_metrics
from utils.checkpoint_utils import trainable_state_dict, load_any_state_dict

CSV_PATH = os.path.join(BASE_DIR, "data", "osv5m_serbia_metadata.csv")
# Points at the same top-level raw_images folder GeoToken's own colab
# scripts use (build_gallery_colab.py's DRIVE_RAW_IMAGES) rather than the
# nested data/raw_images/ copy, which has already drifted (20,389 files vs.
# the metadata CSV's 20,366 rows) — one canonical source, not two copies.
# Locally (no /content/drive in BASE_DIR) this constant is unused anyway,
# since sync_images_to_local() no-ops and callers should pass a local path.
DRIVE_RAW_IMAGES_DIR = "/content/drive/MyDrive/raw_images" if "/content/drive" in BASE_DIR \
    else os.path.join(BASE_DIR, "data", "raw_images")
LOCAL_RAW_IMAGES_DIR = "/content/raw_images"
SPLITS_DIR = os.path.join(BASE_DIR, "data", "processed", "splits")


def sync_images_to_local(src_dir, dest_dir):
    """Copies raw_images to local Colab disk once, so every subsequent epoch
    reads from fast local disk instead of the Drive mount — reading ~16k
    small files per epoch directly off Drive is what made the first
    v1_full_unfreeze run take ~6.2s/iteration (~13h for 15 epochs). No-op on
    a machine that isn't Colab (BASE_DIR not under /content/drive) —
    returns src_dir unchanged so local smoke tests keep working as before.

    Prefers copying+extracting a sibling raw_images.zip (one Drive file
    transfer, same as build_gallery_colab.py's earlier manual workaround)
    over a file-by-file copy loop, which pays Drive-mount per-file latency
    ~16,000 times instead of once."""
    if "/content/drive" not in src_dir:
        return src_dir
    if os.path.isdir(dest_dir) and len(os.listdir(dest_dir)) > 0:
        print(f"[SYNC] {dest_dir} already has {len(os.listdir(dest_dir))} files. Skipping.")
        return dest_dir

    zip_path = src_dir.rstrip("/") + ".zip"
    if os.path.exists(zip_path):
        local_zip = "/content/raw_images.zip"
        print(f"[SYNC] Copying {zip_path} -> {local_zip} ...")
        shutil.copy2(zip_path, local_zip)
        extract_root = os.path.dirname(dest_dir.rstrip("/")) or "/content"
        print(f"[SYNC] Extracting to {extract_root} ...")
        subprocess.run(["unzip", "-q", local_zip, "-d", extract_root], check=True)
        print(f"[SYNC] Done. {len(os.listdir(dest_dir))} files now in {dest_dir}")
        return dest_dir

    print(f"[SYNC] No {zip_path} found — falling back to file-by-file copy (slower).")
    os.makedirs(dest_dir, exist_ok=True)
    src_files = set(os.listdir(src_dir))
    existing = set(os.listdir(dest_dir))
    to_copy = src_files - existing
    if not to_copy:
        print(f"[SYNC] {dest_dir} already has all {len(src_files)} files. Skipping copy.")
        return dest_dir
    print(f"[SYNC] Copying {len(to_copy)} files {src_dir} -> {dest_dir} (one-time cost)...")
    for fname in tqdm(to_copy, desc="Syncing images"):
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dest_dir, fname))
    print(f"[SYNC] Done. {len(os.listdir(dest_dir))} files now in {dest_dir}")
    return dest_dir


def sanity_check_images(df, img_dir, n_check=20):
    """Fail fast instead of silently training on blank placeholder images —
    SerbiaGeoCLIPDataset falls back to a blank black image for any id it
    can't find on disk, with no warning printed."""
    found = 0
    for _, row in df.head(n_check).iterrows():
        if os.path.exists(os.path.join(img_dir, f"{row['id']}.jpg")):
            found += 1
    if found == 0:
        print(f"\n[FATAL] 0/{n_check} sample images resolved in: {img_dir}")
        sys.exit(1)
    print(f"[OK] Pre-flight check: {found}/{n_check} sample images resolved in {img_dir}")


def compute_loss(pred_coords, true_coords, bbox_min, bbox_max):
    pred_norm = (pred_coords - bbox_min) / (bbox_max - bbox_min)
    true_norm = (true_coords - bbox_min) / (bbox_max - bbox_min)
    l1 = F.l1_loss(pred_norm, true_norm)
    hav_km = haversine_distance(pred_coords, true_coords).mean()
    loss = 0.05 * l1 + 0.95 * (hav_km / 1000.0)
    return loss, l1.item(), hav_km.item()


def build_scheduler(optimizer, scheduler_name, total_steps, warmup_steps, eta_min):
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=eta_min
        )

    def warmup_cosine_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)))
        base_lr = optimizer.defaults["lr"]
        floor = eta_min / base_lr
        return floor + (1 - floor) * cosine.item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)


@torch.no_grad()
def evaluate(model, loader, device, bbox_min, bbox_max):
    model.eval()
    all_pred, all_true = [], []
    total_loss = 0.0
    n_batches = 0
    for pixel_values, coords in loader:
        pixel_values, coords = pixel_values.to(device), coords.to(device)
        # Autocast scoped to just the forward pass, not the loss — haversine's
        # sin/cos/sqrt/atan2 chain loses precision (risk of sqrt of a
        # slightly-negative value -> NaN) if left running in fp16.
        with autocast(enabled=(device == "cuda")):
            pred = model(pixel_values)
        pred = pred.float()
        loss, _, _ = compute_loss(pred, coords, bbox_min, bbox_max)
        total_loss += loss.item()
        n_batches += 1
        all_pred.append(pred.cpu())
        all_true.append(coords.cpu())
    model.train()

    pred_coords = torch.cat(all_pred, dim=0)
    true_coords = torch.cat(all_true, dim=0)
    distances = haversine_distance(pred_coords, true_coords)
    metrics = {}
    for th in (1, 25, 200, 750, 2500):
        metrics[f"acc@{th}km"] = (distances <= th).float().mean().item() * 100
    metrics["median_km"] = distances.median().item()
    metrics["mean_km"] = distances.mean().item()
    metrics["val_loss"] = total_loss / max(1, n_batches)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--adaptation", choices=["lora", "full_unfreeze"], default="lora")
    parser.add_argument("--lora_targets", type=str, default="q,k,v,out")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--scheduler", choices=["cosine", "warmup_cosine"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a last_model.pt to resume from. Model/optimizer/scaler "
             "state is restored; the scheduler is rebuilt fresh, scoped to "
             "the remaining epochs (CosineAnnealingLR is periodic past "
             "T_max, so naively continuing the old schedule would jump back "
             "up toward the base LR instead of continuing to decay). "
             "--adaptation/--lora_* /--unfreeze_last_n_layers must match the "
             "original run — the architecture isn't recovered from the "
             "checkpoint, only its weights.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print(f"TRAINING GeoCLIP-Regression — run: {args.run_name}")
    print("=" * 65)
    print("Hyperparameters:")
    for key, value in vars(args).items():
        if key == "run_name":
            continue
        print(f"  {key:<22}: {value}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    checkpoint_dir = os.path.join(BASE_DIR, "geoclip_adaptation", "checkpoints", args.run_name)
    tb_log_dir = os.path.join(BASE_DIR, "geoclip_adaptation", "tensorboard_logs", args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    lora_targets = tuple(t.strip() for t in args.lora_targets.split(","))
    model = GeoCLIPRegressor(
        adaptation=args.adaptation,
        lora_targets=lora_targets,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
    ).to(device)
    print(f"Trainable params: {model.trainable_parameter_count():,}")

    raw_images_dir = sync_images_to_local(DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR)

    train_ds = SerbiaGeoCLIPDataset(
        CSV_PATH, raw_images_dir, os.path.join(SPLITS_DIR, "train_indices.pt"), train=True
    )
    val_ds = SerbiaGeoCLIPDataset(
        CSV_PATH, raw_images_dir, os.path.join(SPLITS_DIR, "val_indices.pt"), train=False
    )
    sanity_check_images(train_ds.df, raw_images_dir)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    use_amp = (device == "cuda")
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 1
    best_val_median_km = float("inf")
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        load_any_state_dict(model, ckpt)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_median_km = ckpt["best_val_median_km"]
        print(f"  Resuming at epoch {start_epoch}, best-so-far: {best_val_median_km:.1f}km")
        if start_epoch > args.epochs:
            print(f"[ERROR] --epochs ({args.epochs}) must exceed the resumed "
                  f"checkpoint's epoch ({ckpt['epoch']}).")
            sys.exit(1)

    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    # Scoped to the epochs actually remaining, not args.epochs total — a
    # scheduler resumed with the ORIGINAL total_steps would restart mid-cycle
    # instead of continuing to decay (CosineAnnealingLR climbs back toward
    # base_lr once past its original T_max).
    remaining_epochs = args.epochs - start_epoch + 1
    total_steps = steps_per_epoch * remaining_epochs
    scheduler = build_scheduler(
        optimizer, args.scheduler, total_steps, args.warmup_steps, args.eta_min
    )

    bbox_min = model.bbox_min.to(device)
    bbox_max = model.bbox_max.to(device)

    csv_path = os.path.join(checkpoint_dir, "training_log.csv")
    csv_fields = [
        "epoch", "train_loss", "train_l1", "train_hav_km",
        "val_loss", "val_median_km", "val_mean_km",
        "acc@1km", "acc@25km", "acc@200km", "acc@750km", "acc@2500km", "lr",
    ]
    if not args.resume:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_fields)

    global_step = 0
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled (no GPU)'}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss, running_l1, running_hav = 0.0, 0.0, 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}")
        for i, (pixel_values, coords) in enumerate(pbar):
            pixel_values, coords = pixel_values.to(device), coords.to(device)
            with autocast(enabled=use_amp):
                pred = model(pixel_values)
            # Loss stays in fp32 — see evaluate()'s comment on haversine's
            # sin/cos/sqrt/atan2 chain and fp16 precision.
            pred = pred.float()
            loss, l1_val, hav_val = compute_loss(pred, coords, bbox_min, bbox_max)
            scaler.scale(loss / args.grad_accum_steps).backward()

            if (i + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            running_loss += loss.item()
            running_l1 += l1_val
            running_hav += hav_val
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", hav_km=f"{hav_val:.1f}")

        train_loss = running_loss / n_batches
        train_l1 = running_l1 / n_batches
        train_hav = running_hav / n_batches
        current_lr = optimizer.param_groups[0]["lr"]

        val_metrics = evaluate(model, val_loader, device, bbox_min, bbox_max)

        print(
            f" Epoch {epoch:02d} [TRAIN] loss={train_loss:.4f} hav={train_hav:.1f}km | "
            f"[VAL] {format_metrics(val_metrics)} | loss={val_metrics['val_loss']:.4f}"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
        writer.add_scalar("Haversine_km/train", train_hav, epoch)
        writer.add_scalar("Haversine_km/val_median", val_metrics["median_km"], epoch)
        writer.add_scalar("LR", current_lr, epoch)
        for th in (1, 25, 200, 750, 2500):
            writer.add_scalar(f"Accuracy/{th}km", val_metrics[f"acc@{th}km"], epoch)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, train_l1, train_hav,
                val_metrics["val_loss"], val_metrics["median_km"], val_metrics["mean_km"],
                val_metrics["acc@1km"], val_metrics["acc@25km"], val_metrics["acc@200km"],
                val_metrics["acc@750km"], val_metrics["acc@2500km"], current_lr,
            ])

        if val_metrics["median_km"] < best_val_median_km:
            best_val_median_km = val_metrics["median_km"]
            torch.save(trainable_state_dict(model),
                       os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"  [SAVED] New best val median: {best_val_median_km:.1f}km")

        # Overwritten every epoch — full state needed to resume correctly,
        # unlike best_model.pt which is weights-only (evaluation doesn't
        # need optimizer/scaler state, only --resume does).
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
