"""
utils/eval_utils.py — turn a model's predicted S2 tokens into real
lat/lon accuracy numbers (the paper's accuracy@threshold + median error).

Used both for periodic validation checks during training, and for the
final one-time test evaluation.
"""
import time
import torch
from utils.s2_geometry_helpers import s2_tokens_to_coords


def haversine_distance(coord1: torch.Tensor, coord2: torch.Tensor) -> torch.Tensor:
    """
    coord1, coord2: [..., 2] tensors of (lat, lon) in degrees.
    Returns distance in kilometers, shape [...].
    """
    R = 6371.0
    lat1, lon1 = torch.deg2rad(coord1[..., 0]), torch.deg2rad(coord1[..., 1])
    lat2, lon2 = torch.deg2rad(coord2[..., 0]), torch.deg2rad(coord2[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return R * c


@torch.no_grad()
def evaluate(
    model,
    features: dict,
    searcher,
    device: str,
    top_k: int = 15,
    batch_size: int = 64,
    thresholds=(1, 25, 200, 750, 2500),
    verbose: bool = False,
):
    """
    Runs the model in real inference mode (no ground-truth tokens) over an
    entire split and computes haversine-based accuracy.

    Args:
        model: GeoTokenTransformer (must have generate_greedy()).
        features: dict with 'embeddings' [N,768] and 'coords' [N,2] (lat,lon)
                  — e.g. loaded straight from val_features.pt / test_features.pt.
        searcher: GallerySearcher built over the TRAIN gallery. Neighbors are
                  always retrieved from train, even when evaluating val/test —
                  train is the only valid retrieval source.
        device: 'cuda' / 'mps' / 'cpu'.
        top_k: number of retrieved neighbors (should match training, 15).
        thresholds: distance thresholds in km for accuracy@threshold.

    Returns:
        metrics: dict of {'acc@1km': ..., ..., 'median_km': ..., 'mean_km': ...}
        pred_coords: FloatTensor [N, 2] of predicted (lat, lon)
        distances: FloatTensor [N] haversine error per sample, in km
    """
    was_training = model.training
    model.eval()

    embeddings = features["embeddings"]
    true_coords = features["coords"]
    n = embeddings.size(0)

    all_pred_coords = []
    start_time = time.time()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_emb = embeddings[start:end].to(device)

        neighbor_s2, neighbor_embeds, _ = searcher.find_neighbors(
            batch_emb, top_k=top_k, exclude_self=False
        )

        pred_tokens = model.generate_greedy(batch_emb, neighbor_embeds, neighbor_s2)
        pred_tokens_cpu = pred_tokens.cpu().tolist()

        for tok_seq in pred_tokens_cpu:
            lat, lon = s2_tokens_to_coords(tok_seq)
            all_pred_coords.append([lat, lon])

        if verbose:
            print(f"  eval {end}/{n}", end="\r")

    pred_coords = torch.tensor(all_pred_coords, dtype=torch.float32)
    distances = haversine_distance(pred_coords, true_coords)

    metrics = {}
    for th in thresholds:
        metrics[f"acc@{th}km"] = (distances <= th).float().mean().item() * 100
    metrics["median_km"] = distances.median().item()
    metrics["mean_km"] = distances.mean().item()
    metrics["eval_time_sec"] = time.time() - start_time

    if was_training:
        model.train()

    return metrics, pred_coords, distances


def format_metrics(metrics: dict, thresholds=(1, 25, 200, 750, 2500)) -> str:
    """One-line summary matching the paper's table format."""
    parts = [f"{th}km={metrics[f'acc@{th}km']:.2f}%" for th in thresholds]
    return (
        f"median={metrics['median_km']:.1f}km | "
        + " ".join(parts)
    )
