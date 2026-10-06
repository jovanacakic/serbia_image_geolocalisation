import os
import torch

class GallerySearcher:
  def __init__(self, gallery_path="geotoken/context_retrieval/gallery_database.pt", device="cpu"):
    if not os.path.isabs(gallery_path):
      base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
      gallery_path = os.path.join(base_dir, gallery_path)
    gallery_data = torch.load(gallery_path, weights_only=True)
    self.device = device
    self.gallery_embeddings = gallery_data["embeddings"].to(device)
    self.gallery_s2_tokens = gallery_data["s2_tokens"]
    self.gallery_coords = gallery_data["coords"]
    self.gallery_embeddings = self.gallery_embeddings / self.gallery_embeddings.norm(dim=-1, keepdim=True)

  @torch.no_grad()
  def find_neighbors(self, query_embeddings, top_k=15, exclude_self=False):
    query_embeddings = query_embeddings.to(self.device)
    query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)
    sim_matrix = torch.matmul(query_embeddings, self.gallery_embeddings.T)
    k_to_fetch = top_k + 1 if exclude_self else top_k
    scores, topk_indices = torch.topk(sim_matrix, k=k_to_fetch, dim=-1, largest=True)
    if exclude_self:
      topk_indices = topk_indices[:, 1:]
      scores = scores[:, 1:]
    indices_cpu = topk_indices.cpu()
    b_size, k_size = indices_cpu.shape
    # NOTE: when exclude_self=True, the [:, 1:] slice above produces a
    # non-contiguous tensor, which .view(-1) cannot handle. .reshape(-1)
    # does (copies under the hood when needed) — .view() alone would raise
    # "view size is not compatible with input tensor's size and stride"
    # on every single training call, since training always uses exclude_self=True.
    flat_indices = indices_cpu.reshape(-1)
    neighbor_tokens = self.gallery_s2_tokens[flat_indices].view(b_size, k_size, -1)
    neighbor_embeddings = self.gallery_embeddings[flat_indices].view(b_size, k_size, -1)
    return neighbor_tokens.to(self.device), neighbor_embeddings.to(self.device), scores
