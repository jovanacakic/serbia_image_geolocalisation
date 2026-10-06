import os
import sys
import csv
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root directory to path (two levels up: transformer_model/ -> geotoken/ -> root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from geotoken.context_retrieval.search_neighbors import GallerySearcher
from geotoken.transformer_model.transformer_architecture import GeoTokenTransformer
from utils.eval_utils import evaluate, format_metrics


class PrecomputedGeoDataset(Dataset):
  """Dataset utilizing in-memory pre-extracted embeddings and S2 tokens."""

  def __init__(
      self,
      embeddings: torch.Tensor,
      s2_tokens: torch.Tensor,
      searcher: GallerySearcher,
      top_k: int = 15,
  ):
    self.embeddings = embeddings
    self.s2_tokens = s2_tokens
    self.searcher = searcher
    self.top_k = top_k

  def __len__(self):
    return len(self.embeddings)

  def __getitem__(self, idx):
    query_emb = self.embeddings[idx]
    target_token = self.s2_tokens[idx]
    return query_emb, target_token, idx


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--run_name", type=str, default="baseline",
      help="Names this run — keeps checkpoints and TensorBoard logs separate "
           "across ablation runs (e.g. 'no_retrieval', 'beam_search', "
           "'level15'), and lets TensorBoard overlay multiple runs for "
           "comparison instead of overwriting the previous run's files.",
  )
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--weight_decay", type=float, default=1e-2)
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--top_k", type=int, default=15)
  parser.add_argument("--num_encoder_layers", type=int, default=6)
  parser.add_argument("--num_decoder_layers", type=int, default=6)
  parser.add_argument("--dim_feedforward", type=int, default=2048)
  parser.add_argument(
      "--scheduler", type=str, default="cosine", choices=["cosine", "step"],
      help="'cosine' = CosineAnnealingLR (this project's original default). "
           "'step' = StepLR, matching the original GeoToken repo's "
           "recipe (decays lr by --step_gamma every --step_size epochs).",
  )
  parser.add_argument("--step_size", type=int, default=1)
  parser.add_argument("--step_gamma", type=float, default=0.9)
  parser.add_argument(
      "--loss_weighting", type=str, default="uniform",
      choices=["uniform", "paper"],
      help="'uniform' (this project's original default): face gets full "
           "CE weight, the 20 quadrant CEs are averaged, then summed with "
           "face — face ends up ~20x the per-token weight of any single "
           "quadrant level. 'paper': matches the original GeoToken repo — "
           "a linearly-declining per-position weight (2.0 at the coarse "
           "face token, down to 1.0 at the finest quadrant), normalized by "
           "the sum of weights.",
  )
  parser.add_argument(
      "--resume", type=str, default=None,
      help="Path to a checkpoint (e.g. last_geotoken_model.pt) to continue "
           "training from, instead of starting at epoch 1. Architecture "
           "flags (--num_encoder_layers/--num_decoder_layers/"
           "--dim_feedforward/--dropout) must match the checkpoint being "
           "resumed. --lr becomes the fresh starting lr for the resumed "
           "segment (the old schedule's decayed lr is not reused).",
  )
  parser.add_argument(
      "--gallery_path", type=str, default=None,
      help="Override the retrieval gallery .pt file (relative to project "
           "root, or absolute). Defaults to "
           "geotoken/context_retrieval/gallery_database.pt (train only). "
           "Use this to point at a combined train+val gallery for a final "
           "retrain once model/hyperparameter decisions are locked in.",
  )
  parser.add_argument(
      "--val_features_path", type=str, default=None,
      help="Override the val feature file. Must be the same embedding type "
           "as --gallery_path: pairing a geo-aligned gallery with raw-CLIP "
           "val embeddings would make retrieval meaningless.",
  )
  parser.add_argument(
      "--clip_dim", type=int, default=768,
      help="Image embedding dimension in the feature files. 768 for raw "
           "CLIP; 1280 for the geo-aligned features (raw 768 + G3-aligned "
           "512) written by "
           "geotoken/context_retrieval/extract_geoaligned_features.py.",
  )
  parser.add_argument(
      "--no_val_eval", action="store_true",
      help="Skip loading val_features.pt and the per-epoch evaluate() call "
           "entirely. Use this when --gallery_path points at a combined "
           "train+val gallery — val is no longer legitimately held out in "
           "that case, so there's no data left to validate against. Trains "
           "for the fixed --epochs given and saves last_geotoken_model.pt "
           "each epoch; that final checkpoint is the artifact to evaluate "
           "on test. No best_geotoken_model.pt selection happens in this "
           "mode, since there's no val signal to select by.",
  )
  args = parser.parse_args()

  print("=" * 65)
  print(f"TRAINING GeoToken Transformer — run: {args.run_name}")
  print("=" * 65)
  print("Hyperparameters:")
  for key, value in vars(args).items():
    if key == "run_name":
      continue
    print(f"  {key:<18}: {value}")
  print("=" * 65)

  BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  if args.gallery_path is None:
    GALLERY_PATH = os.path.join(BASE_DIR, "geotoken", "context_retrieval", "gallery_database.pt")
  elif os.path.isabs(args.gallery_path):
    GALLERY_PATH = args.gallery_path
  else:
    GALLERY_PATH = os.path.join(BASE_DIR, args.gallery_path)
  if args.val_features_path is None:
    VAL_FEATURES_PATH = os.path.join(BASE_DIR, "geotoken", "context_retrieval", "val_features.pt")
  elif os.path.isabs(args.val_features_path):
    VAL_FEATURES_PATH = args.val_features_path
  else:
    VAL_FEATURES_PATH = os.path.join(BASE_DIR, args.val_features_path)
  CHECKPOINT_DIR = os.path.join(
      BASE_DIR, "geotoken", "transformer_model", "checkpoints", args.run_name
  )
  TB_LOG_DIR = os.path.join(
      BASE_DIR, "geotoken", "transformer_model", "tensorboard_logs", args.run_name
  )
  LOG_PATH = os.path.join(CHECKPOINT_DIR, "training_log.csv")
  os.makedirs(CHECKPOINT_DIR, exist_ok=True)
  os.makedirs(TB_LOG_DIR, exist_ok=True)

  print(f"Checkpoints      : {CHECKPOINT_DIR}")
  print(f"TensorBoard logs : {TB_LOG_DIR}")

  writer = SummaryWriter(log_dir=TB_LOG_DIR)
  writer.add_text("config", str(vars(args)))

  # Hardware selection
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Training on device: {device}")

  # 1. Load Gallery and Retriever. NOTE: the train gallery is the ONLY valid
  # retrieval source — used both as training queries (with exclude_self=True,
  # since a training image would otherwise trivially retrieve itself) and,
  # later, as the neighbor source when evaluating val/test (exclude_self=False,
  # since val/test images genuinely aren't in this gallery).
  searcher = GallerySearcher(gallery_path=GALLERY_PATH, device=device)
  gallery_data = torch.load(GALLERY_PATH, weights_only=True)

  embeddings = gallery_data["embeddings"]  # [N, 768]
  s2_tokens = gallery_data["s2_tokens"]  # [N, 21]

  # 1b. Load val features (queries only — retrieval still comes from `searcher`,
  # i.e. the train gallery, never from val itself). Skipped entirely under
  # --no_val_eval, since that mode is for a final retrain on a combined
  # train+val gallery — val is no longer legitimately held out at that point.
  val_data = None
  if not args.no_val_eval:
    assert os.path.exists(VAL_FEATURES_PATH), (
        f"Missing {VAL_FEATURES_PATH}. Run extract_split_features(_colab).py "
        f"--split val first."
    )
    val_data = torch.load(VAL_FEATURES_PATH, weights_only=True)
    print(f"Loaded val set for evaluation: {val_data['embeddings'].shape[0]} images")
  else:
    print("--no_val_eval set: skipping val load and per-epoch evaluation.")

  dataset = PrecomputedGeoDataset(
      embeddings, s2_tokens, searcher, top_k=args.top_k
  )
  train_loader = DataLoader(
      dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
  )

  # 2. Model, Loss, Optimizer
  model = GeoTokenTransformer(
      clip_dim=args.clip_dim,
      d_model=512,
      nhead=8,
      num_encoder_layers=args.num_encoder_layers,
      num_decoder_layers=args.num_decoder_layers,
      dim_feedforward=args.dim_feedforward,
      dropout=args.dropout,
  ).to(device)

  # 3. Training Loop — NUM_EPOCHS defined here so the scheduler below can
  # reference it directly, keeping them permanently in sync (this is what
  # caused the earlier T_max=15/NUM_EPOCHS=15 coupling risk).
  NUM_EPOCHS = args.epochs
  THRESHOLDS = (1, 25, 200, 750, 2500)

  criterion = nn.CrossEntropyLoss()
  position_weights = None
  if args.loss_weighting == "paper":
    # 21 positions: face (index 0) + 20 quadrant levels. Matches the
    # original GeoToken repo's model_transformer_add_c_n_grouped.py exactly.
    position_weights = torch.linspace(2.0, 1.0, steps=21).to(device)
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay
  )

  log_fields = (
      ["epoch", "train_loss", "train_face_acc", "train_quad_acc",
       "val_median_km", "val_mean_km"]
      + [f"val_acc@{t}km" for t in THRESHOLDS]
      + ["val_eval_time_sec"]
  )

  # Resume: load model/optimizer state and continue the epoch count, rather
  # than starting over. The old schedule's decayed lr is NOT reused — it
  # would be near `eta_min` by the end of a cosine run, so continuing with it
  # verbatim would mean no further real learning. --lr instead becomes the
  # fresh starting point for this resumed segment.
  start_epoch = 1
  best_val_median_km = float("inf")
  if args.resume:
    assert os.path.exists(args.resume), f"--resume checkpoint not found: {args.resume}"
    print(f"Resuming from checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    for group in optimizer.param_groups:
      group["lr"] = args.lr
    start_epoch = ckpt["epoch"] + 1
    resumed_val = ckpt.get("val_metrics")
    resumed_val_str = f"{resumed_val['median_km']:.1f}km" if resumed_val else "n/a (checkpoint saved under --no_val_eval)"
    print(f"  Resumed model/optimizer from epoch {ckpt['epoch']} "
          f"(val median {resumed_val_str}).")
    print(f"  Continuing epochs {start_epoch}-{NUM_EPOCHS}, lr reset to {args.lr}.")
    if os.path.exists(LOG_PATH):
      with open(LOG_PATH, "r", newline="") as f:
        prior_rows = list(csv.DictReader(f))
      if prior_rows:
        best_val_median_km = min(float(r["val_median_km"]) for r in prior_rows)
        print(f"  Best val median so far (from training_log.csv): "
              f"{best_val_median_km:.1f}km")

  # Only truncate + write a fresh header when NOT continuing an existing log.
  if not (args.resume and os.path.exists(LOG_PATH)):
    with open(LOG_PATH, "w", newline="") as f:
      csv.writer(f).writerow(log_fields)

  # Cosine T_max spans only the epochs actually left to run, so a resumed
  # segment anneals cleanly from its own fresh --lr down to eta_min, instead
  # of restarting the original curve or (worse) continuing past T_max, where
  # CosineAnnealingLR is periodic and lr would climb back toward the
  # original base lr rather than staying low.
  remaining_epochs = NUM_EPOCHS - start_epoch + 1
  if args.scheduler == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining_epochs, eta_min=1e-6
    )
  else:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.step_gamma
    )

  for epoch in range(start_epoch, NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0.0
    total_face_acc = 0.0
    total_quad_acc = 0.0
    num_batches = 0

    pbar = tqdm(
        train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS:02d}"
    )

    for q_emb, target_toks, _ in pbar:
      q_emb = q_emb.to(device)
      target_toks = target_toks.to(device)

      # Dynamic RAG search: retrieve top-15 neighbors excluding self (rank 0)
      with torch.no_grad():
        neighbor_s2, neighbor_embeds, _ = searcher.find_neighbors(
            q_emb, top_k=args.top_k, exclude_self=True
        )

      optimizer.zero_grad()

      # Forward pass (teacher forcing — training only, never for real inference)
      logits_face, logits_quads = model(
          q_emb, neighbor_embeds, neighbor_s2, target_toks
      )

      # 1. Level 0 Loss (Cube Face: 6 classes)
      loss_face = criterion(logits_face, target_toks[:, 0])

      # 2. Levels 1..20 Loss (Quadrants: 4 classes each)
      quad_losses = []
      quad_correct = 0
      total_quad_cells = 0

      for lvl in range(20):
        target_lvl = target_toks[:, lvl + 1]
        loss_lvl = criterion(logits_quads[lvl], target_lvl)
        quad_losses.append(loss_lvl)

        pred_lvl = logits_quads[lvl].argmax(dim=-1)
        quad_correct += (pred_lvl == target_lvl).sum().item()
        total_quad_cells += target_lvl.numel()

      if args.loss_weighting == "paper":
        position_losses = [loss_face] + quad_losses
        loss = sum(w * l for w, l in zip(position_weights, position_losses))
        loss = loss / position_weights.sum()
      else:
        # Average quadrant loss across the 20 levels, then sum with face.
        loss_quads = sum(quad_losses) / 20.0
        loss = loss_face + loss_quads

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      # Metrics
      face_correct = (
          (logits_face.argmax(dim=-1) == target_toks[:, 0])
          .sum()
          .item()
      )
      face_acc = face_correct / target_toks.size(0)
      quad_acc = quad_correct / total_quad_cells

      total_loss += loss.item()
      total_face_acc += face_acc
      total_quad_acc += quad_acc
      num_batches += 1

      pbar.set_postfix({
          "Loss": f"{loss.item():.4f}",
          "FaceAcc": f"{face_acc*100:.1f}%",
          "QuadAcc": f"{quad_acc*100:.1f}%",
      })

    scheduler.step()

    avg_loss = total_loss / num_batches
    avg_face_acc = (total_face_acc / num_batches) * 100
    avg_quad_acc = (total_quad_acc / num_batches) * 100

    print(
        f" Epoch {epoch:02d} [TRAIN] Loss={avg_loss:.4f} | "
        f"FaceAcc={avg_face_acc:.2f}% | QuadAcc={avg_quad_acc:.2f}%"
    )

    # --- Real validation: greedy-decode the val set, no ground truth used,
    # exactly what a genuinely new image would go through. Skipped entirely
    # under --no_val_eval (val is folded into the training gallery in that
    # mode, so there's no legitimate held-out data left to evaluate). ---
    if not args.no_val_eval:
      val_metrics, _, _ = evaluate(
          model, val_data, searcher, device, top_k=args.top_k,
          batch_size=args.batch_size, thresholds=THRESHOLDS,
      )
      print(f" Epoch {epoch:02d} [VAL] {format_metrics(val_metrics, THRESHOLDS)} "
            f"(eval took {val_metrics['eval_time_sec']:.1f}s)")

      with open(LOG_PATH, "a", newline="") as f:
        row = (
            [epoch, avg_loss, avg_face_acc, avg_quad_acc,
             val_metrics["median_km"], val_metrics["mean_km"]]
            + [val_metrics[f"acc@{t}km"] for t in THRESHOLDS]
            + [val_metrics["eval_time_sec"]]
        )
        csv.writer(f).writerow(row)

      writer.add_scalar("val/median_km", val_metrics["median_km"], epoch)
      writer.add_scalar("val/mean_km", val_metrics["mean_km"], epoch)
      for th in THRESHOLDS:
        writer.add_scalar(f"val/acc@{th}km", val_metrics[f"acc@{th}km"], epoch)
    else:
      val_metrics = None
      # Keep the CSV column count consistent with log_fields' header even
      # with no val columns to report, rather than writing a mismatched row.
      with open(LOG_PATH, "a", newline="") as f:
        row = [epoch, avg_loss, avg_face_acc, avg_quad_acc] + [""] * (len(log_fields) - 4)
        csv.writer(f).writerow(row)

    # Train metrics, logged to TensorBoard — this is what lets you overlay
    # multiple ablation runs (different --run_name values) on one graph
    # later, instead of comparing static screenshots by eye.
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/face_acc", avg_face_acc, epoch)
    writer.add_scalar("train/quad_acc", avg_quad_acc, epoch)

    # Overwrite a single "last" checkpoint each epoch (for resuming), instead
    # of accumulating one file per epoch — training_log.csv and TensorBoard
    # already keep the full per-epoch metric history, so per-epoch weights
    # are the part that's rarely needed and costs the most disk space.
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_loss,
            "val_metrics": val_metrics,
        },
        os.path.join(CHECKPOINT_DIR, "last_geotoken_model.pt"),
    )

    # "Best" is chosen by VAL performance, not train loss. Train loss keeps
    # dropping even once the model starts overfitting to the train gallery's
    # retrieval crutch; val median error is what actually reflects
    # generalization to images the model hasn't seen. No val_metrics under
    # --no_val_eval, so no best-checkpoint selection happens in that mode —
    # last_geotoken_model.pt at the final epoch is the artifact to use.
    if val_metrics is not None and val_metrics["median_km"] < best_val_median_km:
      best_val_median_km = val_metrics["median_km"]
      torch.save(
          model.state_dict(),
          os.path.join(CHECKPOINT_DIR, "best_geotoken_model.pt"),
      )
      print(f" Saved new best model checkpoint "
            f"(val median error: {best_val_median_km:.1f}km)")

    epoch_duration = time.time() - epoch_start_time
    # print(f" Epoch {epoch:02d} took {epoch_duration:.1f}s "
    #       f"({epoch_duration/60:.1f} min)")
    if epoch == start_epoch:
      projected_total_min = (epoch_duration * remaining_epochs) / 60.0
      print(f" [ESTIMATE] Based on epoch {start_epoch}: ~{projected_total_min:.1f} "
            f"min projected for the remaining {remaining_epochs} epochs "
            f"({start_epoch}-{NUM_EPOCHS}).")

  writer.close()

  print("\n" + "=" * 65)
  if args.no_val_eval:
    print("[TRAINING COMPLETE] --no_val_eval mode: no val-based best-checkpoint "
          "selection was done.")
    print(f"Final checkpoint : {CHECKPOINT_DIR}/last_geotoken_model.pt")
  else:
    print(f"[TRAINING COMPLETE] Best val median error: {best_val_median_km:.1f}km")
    print(f"Best checkpoint : {CHECKPOINT_DIR}/best_geotoken_model.pt")
  print(f"Training log    : {LOG_PATH}")
  print("Test set was NOT touched — evaluate it once, at the very end, "
        "after all model/config decisions are locked in.")
  print("=" * 65)


if __name__ == "__main__":
  main()
