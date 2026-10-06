"""
geotoken/transformer_model/evaluate_pool.py — evaluate a trained checkpoint
under temperature-sampling + log-probability selection, alongside the
existing greedy baseline and the "Ideal Selector" oracle diagnostic
(paper Table III). Val only — the test set stays untouched until every
model/config decision, including decoding strategy, is finalized.

Usage (Colab):
  !python geotoken/transformer_model/evaluate_pool.py \
    --checkpoint geotoken/transformer_model/checkpoints/loss_paper_plus_reg/best_geotoken_model.pt \
    --pool_size 10 --temperature 0.7
"""
import os
import sys
import csv
import time
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from geotoken.context_retrieval.search_neighbors import GallerySearcher
from geotoken.transformer_model.transformer_architecture import GeoTokenTransformer
from utils.eval_utils import haversine_distance, format_metrics
from utils.s2_geometry_helpers import s2_tokens_to_coords


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--checkpoint", type=str, required=True,
      help="Path to a checkpoint to evaluate — either a plain state_dict "
           "(best_geotoken_model.pt) or a full dict with a "
           "'model_state_dict' key (last_geotoken_model.pt).",
  )
  parser.add_argument(
      "--gallery_path", type=str,
      default="geotoken/context_retrieval/gallery_database.pt",
      help="Retrieval gallery. Must match the checkpoint's embedding type.")
  parser.add_argument(
      "--val_features_path", type=str,
      default="geotoken/context_retrieval/val_features.pt",
      help="Val features. Must match --gallery_path's embedding type.")
  parser.add_argument(
      "--clip_dim", type=int, default=768,
      help="768 for raw CLIP features, 1280 for geo-aligned.")
  parser.add_argument(
      "--g3_checkpoint", type=str, default=None,
      help="Enables similarity-based selection (GeoToken paper's third "
           "strategy, i.e. G3 geo-verification applied to GeoToken's "
           "candidate pool). Each candidate coordinate is scored by the "
           "cosine similarity between the query image embedding and that "
           "coordinate's GPS embedding in the trained G3 space.")
  parser.add_argument("--g3_image_adaptation", choices=["frozen_mlp", "lora"],
                      default="frozen_mlp")
  parser.add_argument("--g3_no_text", action="store_true")
  parser.add_argument("--pool_size", type=int, default=10)
  parser.add_argument("--temperature", type=float, default=0.7)
  parser.add_argument("--top_k", type=int, default=15)
  parser.add_argument(
      "--batch_size", type=int, default=32,
      help="Query images per batch. The actual decode batch is this times "
           "--pool_size, so keep it smaller than train.py's batch_size to "
           "avoid OOM.",
  )
  parser.add_argument("--num_encoder_layers", type=int, default=6)
  parser.add_argument("--num_decoder_layers", type=int, default=6)
  parser.add_argument("--dim_feedforward", type=int, default=2048)
  parser.add_argument(
      "--dropout", type=float, default=0.1,
      help="Must match the checkpoint's architecture (shapes only — "
           "dropout itself is inactive in eval mode).",
  )
  parser.add_argument(
      "--output_csv", type=str,
      default="geotoken/transformer_model/pool_eval_results.csv",
      help="Every run appends 3 rows here (Greedy/Log-Probability/Ideal "
           "Selector), tagged with checkpoint/pool_size/temperature, so "
           "results across multiple runs (different K, different "
           "checkpoints) accumulate into one comparable table instead of "
           "living only in scrollback.",
  )
  args = parser.parse_args()

  print("=" * 65)
  print("[POOL EVALUATION] Temperature Sampling + Log-Probability Selection")
  print("=" * 65)
  print(f"Checkpoint  : {args.checkpoint}")
  print(f"Pool size K : {args.pool_size}")
  print(f"Temperature : {args.temperature}")
  print(f"top_k       : {args.top_k}")
  print("=" * 65)

  BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  GALLERY_PATH = args.gallery_path if os.path.isabs(args.gallery_path) \
      else os.path.join(BASE_DIR, args.gallery_path)
  VAL_FEATURES_PATH = args.val_features_path if os.path.isabs(args.val_features_path) \
      else os.path.join(BASE_DIR, args.val_features_path)

  # Resolve --checkpoint relative to the project root (BASE_DIR), not the
  # shell's current working directory — matches how every other path in
  # this codebase (GALLERY_PATH, VAL_FEATURES_PATH, CHECKPOINT_DIR in
  # train.py, etc.) is already built from __file__, so a relative
  # --checkpoint works the same regardless of where `!python ...` is
  # invoked from in a Colab notebook (e.g. /content vs. the project dir).
  checkpoint_path = args.checkpoint
  if not os.path.isabs(checkpoint_path):
    checkpoint_path = os.path.join(BASE_DIR, checkpoint_path)
  assert os.path.exists(checkpoint_path), f"--checkpoint not found: {checkpoint_path}"
  print(f"Resolved to : {checkpoint_path}")

  output_csv_path = args.output_csv
  if not os.path.isabs(output_csv_path):
    output_csv_path = os.path.join(BASE_DIR, output_csv_path)
  os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Running on device: {device}")

  searcher = GallerySearcher(gallery_path=GALLERY_PATH, device=device)


  val_data = torch.load(VAL_FEATURES_PATH, weights_only=True)
  embeddings = val_data["embeddings"]
  true_coords = val_data["coords"]
  n = embeddings.size(0)
  print(f"Loaded val set: {n} images")

  # Similarity-based selection needs the G3 model itself (to embed GPS
  # candidates) and the query images (to embed the query). GeoToken's
  # own feature file stores only precomputed embeddings, so the images
  # are re-encoded here through G3's image tower.
  g3_model, query_img_emb = None, None
  if args.g3_checkpoint:
    import pandas as pd
    from torch.amp import autocast
    from torch.utils.data import DataLoader
    from g3_adaptation.model import G3GeoAlignment
    from geoclip_adaptation.train_lora import (
        sync_images_to_local, DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR)
    from geotoken.context_retrieval.extract_geoaligned_features import IndexedImageDataset
    from utils.checkpoint_utils import load_any_state_dict

    g3_ckpt = args.g3_checkpoint if os.path.isabs(args.g3_checkpoint) \
        else os.path.join(BASE_DIR, args.g3_checkpoint)
    print(f'Loading G3 scorer: {g3_ckpt}')
    g3_model = G3GeoAlignment(image_adaptation=args.g3_image_adaptation,
                              use_text=not args.g3_no_text).to(device)
    load_any_state_dict(g3_model, torch.load(g3_ckpt, map_location=device,
                                             weights_only=True))
    g3_model.eval()

    img_dir = sync_images_to_local(DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR)
    df_all = pd.read_csv(os.path.join(BASE_DIR, 'data', 'osv5m_serbia_metadata.csv'))
    df_val = df_all.iloc[val_data['split_indices'].tolist()]
    loader = DataLoader(IndexedImageDataset(df_val, img_dir), batch_size=64,
                        shuffle=False, num_workers=2)
    embs = []
    with torch.no_grad():
      for px in loader:
        with autocast('cuda', enabled=(device == 'cuda')):
          embs.append(g3_model.encode_image(px.to(device)).float().cpu())
    query_img_emb = torch.cat(embs, dim=0)
    assert query_img_emb.size(0) == n, 'query embedding count != val size'
    print(f'  query image embeddings: {tuple(query_img_emb.shape)}')

  model = GeoTokenTransformer(
      clip_dim=args.clip_dim,
      d_model=512,
      nhead=8,
      num_encoder_layers=args.num_encoder_layers,
      num_decoder_layers=args.num_decoder_layers,
      dim_feedforward=args.dim_feedforward,
      dropout=args.dropout,
  ).to(device)

  ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
  state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
  model.load_state_dict(state_dict)
  model.eval()

  all_greedy_coords = []
  all_logprob_coords = []
  all_ideal_coords = []
  all_sim_coords = []

  for start in range(0, n, args.batch_size):
    end = min(start + args.batch_size, n)
    batch_emb = embeddings[start:end].to(device)
    b = batch_emb.size(0)

    neighbor_s2, neighbor_embeds, _ = searcher.find_neighbors(
        batch_emb, top_k=args.top_k, exclude_self=False
    )

    # Greedy — same baseline as train.py's per-epoch val eval, included
    # here so it's directly comparable in the same run/table.
    greedy_tokens = model.generate_greedy(batch_emb, neighbor_embeds, neighbor_s2)
    for tok_seq in greedy_tokens.cpu().tolist():
      lat, lon = s2_tokens_to_coords(tok_seq)
      all_greedy_coords.append([lat, lon])

    # Sampled candidate pool: [b, K, 21] tokens, [b, K] log-probs.
    pool_tokens, log_probs = model.generate_sample(
        batch_emb, neighbor_embeds, neighbor_s2,
        temperature=args.temperature, num_samples=args.pool_size,
    )
    best_logprob_idx = log_probs.argmax(dim=1)

    true_batch = true_coords[start:end]
    for i in range(b):
      candidate_coords = [
          s2_tokens_to_coords(pool_tokens[i, k].cpu().tolist())
          for k in range(args.pool_size)
      ]

      # Log-probability selection: highest cumulative log-prob in the pool.
      lp_idx = best_logprob_idx[i].item()
      all_logprob_coords.append(list(candidate_coords[lp_idx]))

      # Similarity-based selection: score each candidate coordinate by
      # cosine similarity between the query image embedding and that
      # coordinate's GPS embedding, both in G3's joint space. This is
      # G3's geo-verification stage operating on GeoToken's pool.
      if g3_model is not None:
        cand = torch.tensor(candidate_coords, dtype=torch.float32, device=device)
        with torch.no_grad():
          loc_emb = g3_model.encode_location(cand)
        sims = loc_emb.float() @ query_img_emb[start + i].to(device)
        all_sim_coords.append(list(candidate_coords[sims.argmax().item()]))

      # Ideal Selector (oracle, paper Table III): closest-in-pool to the
      # true coordinate. Not a real predictor — measures how good the
      # candidate pool itself is, independent of the selection strategy.
      true_lat, true_lon = true_batch[i].tolist()
      dists = [
          haversine_distance(
              torch.tensor([lat, lon]), torch.tensor([true_lat, true_lon])
          ).item()
          for lat, lon in candidate_coords
      ]
      best_ideal_idx = min(range(args.pool_size), key=lambda k: dists[k])
      all_ideal_coords.append(list(candidate_coords[best_ideal_idx]))

    print(f"  eval {end}/{n}", end="\r")

  print()

  def summarize(name, pred_coords_list):
    pred_coords = torch.tensor(pred_coords_list, dtype=torch.float32)
    distances = haversine_distance(pred_coords, true_coords)
    metrics = {}
    for th in (1, 25, 200, 750, 2500):
      metrics[f"acc@{th}km"] = (distances <= th).float().mean().item() * 100
    metrics["median_km"] = distances.median().item()
    metrics["mean_km"] = distances.mean().item()
    print(f"{name:20s} {format_metrics(metrics)}")
    return metrics

  print("=" * 65)
  print(f"RESULTS (val set, {n} images, K={args.pool_size}, T={args.temperature})")
  print("=" * 65)
  results = {
      "Greedy": summarize("Greedy (baseline)", all_greedy_coords),
      "Log-Probability": summarize("Log-Probability", all_logprob_coords),
      "Ideal Selector": summarize("Ideal Selector", all_ideal_coords),
  }
  if all_sim_coords:
    results["Similarity-Based"] = summarize("Similarity-Based", all_sim_coords)
  print("=" * 65)

  # Append one row per method to a persistent results CSV, tagged with the
  # run's config, so results from multiple runs (different K, different
  # checkpoints) accumulate into one comparable table over time instead of
  # only existing in terminal scrollback.
  csv_fields = (
      ["timestamp", "checkpoint", "pool_size", "temperature", "top_k",
       "method", "median_km", "mean_km"]
      + [f"acc@{t}km" for t in (1, 25, 200, 750, 2500)]
  )
  write_header = not os.path.exists(output_csv_path)
  with open(output_csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
      writer.writerow(csv_fields)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for method, metrics in results.items():
      row = (
          [timestamp, args.checkpoint, args.pool_size, args.temperature,
           args.top_k, method, metrics["median_km"], metrics["mean_km"]]
          + [metrics[f"acc@{t}km"] for t in (1, 25, 200, 750, 2500)]
      )
      writer.writerow(row)
  print(f"Results appended to: {output_csv_path}")


if __name__ == "__main__":
  main()
