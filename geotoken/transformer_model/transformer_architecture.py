import torch
import torch.nn as nn


class S2HierarchicalEmbedding(nn.Module):
    """Hierarchical embedding for S2 tokens.

    - Token 0: Face index (6 classes -> embedding_dim)
    - Tokens 1-20: Cell quadrants (4 classes per level -> embedding_dim)
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.face_embed = nn.Embedding(6, d_model)
        self.quadrant_embeds = nn.ModuleList(
            [nn.Embedding(4, d_model) for _ in range(20)]
        )
        self.bos_embed = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: LongTensor of shape [B, 21] or [B, K, 21]

        Returns:
            Embeddings of shape [..., 21, d_model]
        """
        orig_shape = tokens.shape[:-1]
        flat_tokens = tokens.view(-1, 21)

        embed_list = []
        embed_list.append(self.face_embed(flat_tokens[:, 0]).unsqueeze(1))
        for lvl in range(1, 21):
            quad_emb = self.quadrant_embeds[lvl - 1](
                flat_tokens[:, lvl]
            ).unsqueeze(1)
            embed_list.append(quad_emb)

        token_embeddings = torch.cat(embed_list, dim=1)
        return token_embeddings.view(*orig_shape, 21, self.d_model)


class GeoTokenTransformer(nn.Module):
    """Transformer Encoder-Decoder for RAG-assisted Geolocation."""

    def __init__(
        self,
        clip_dim: int = 768,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.clip_proj = nn.Linear(clip_dim, d_model)
        self.s2_embedder = S2HierarchicalEmbedding(d_model=d_model)
        self.neighbor_fuse = nn.Linear(d_model * 2, d_model)

        self.encoder_pos = nn.Parameter(torch.randn(1, 1 + 15, d_model) * 0.02)
        self.decoder_pos = nn.Parameter(torch.randn(1, 21, d_model) * 0.02)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.head_face = nn.Linear(d_model, 6)
        self.head_quadrants = nn.ModuleList(
            [nn.Linear(d_model, 4) for _ in range(20)]
        )

    def encode_context(
        self,
        query_clip: torch.Tensor,
        neighbor_clips: torch.Tensor,
        neighbor_s2: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            query_clip: [B, 768]
            neighbor_clips: [B, K, 768]
            neighbor_s2: [B, K, 21]

        Returns:
            Encoder memory tensor: [B, 1 + K, d_model]
        """
        b_size, k_size, _ = neighbor_clips.shape

        q_emb = self.clip_proj(query_clip).unsqueeze(1)
        n_clip_emb = self.clip_proj(neighbor_clips)
        n_s2_emb = self.s2_embedder(neighbor_s2).mean(dim=2)
        n_fused = self.neighbor_fuse(torch.cat([n_clip_emb, n_s2_emb], dim=-1))

        encoder_input = torch.cat([q_emb, n_fused], dim=1)
        encoder_input = encoder_input + self.encoder_pos[:, : 1 + k_size, :]

        memory = self.transformer.encoder(encoder_input)
        return memory

    def forward(
        self,
        query_clip: torch.Tensor,
        neighbor_clips: torch.Tensor,
        neighbor_s2: torch.Tensor,
        target_s2_tokens: torch.Tensor,
    ):
        """Training forward pass with teacher forcing. See module docstring
        of train.py for how this is used — this path REQUIRES ground-truth
        target_s2_tokens and must never be used at inference time, since a
        real query has no ground truth to feed in. Use generate_greedy()
        for inference/evaluation instead.
        """
        b_size = query_clip.size(0)

        memory = self.encode_context(query_clip, neighbor_clips, neighbor_s2)

        tgt_embeddings = self.s2_embedder(target_s2_tokens)
        bos = self.s2_embedder.bos_embed.expand(b_size, 1, -1)
        decoder_input = torch.cat([bos, tgt_embeddings[:, :-1, :]], dim=1)
        decoder_input = decoder_input + self.decoder_pos

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            21, device=query_clip.device
        )

        decoder_out = self.transformer.decoder(
            tgt=decoder_input, memory=memory, tgt_mask=tgt_mask
        )

        logits_face = self.head_face(decoder_out[:, 0, :])

        logits_quadrants = []
        for lvl in range(20):
            logits_lvl = self.head_quadrants[lvl](decoder_out[:, lvl + 1, :])
            logits_quadrants.append(logits_lvl)

        return logits_face, logits_quadrants

    @torch.no_grad()
    def generate_greedy(
        self,
        query_clip: torch.Tensor,
        neighbor_clips: torch.Tensor,
        neighbor_s2: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive greedy decoding for real inference — no ground
        truth tokens available or used. This is what evaluation and any
        real deployment must call, never forward().

        At each step we only know the tokens generated so far, so we embed
        those, run them through the decoder, take the highest-probability
        next token, append it, and repeat for all 21 positions (1 face +
        20 quadrant levels). This mirrors exactly what forward() does
        during teacher forcing, except the "ground truth" fed back in at
        each step is the model's own previous prediction instead of the
        real answer.

        Args:
            query_clip: [B, 768]
            neighbor_clips: [B, K, 768]
            neighbor_s2: [B, K, 21]

        Returns:
            predicted_tokens: LongTensor [B, 21]
        """
        was_training = self.training
        self.eval()

        b_size = query_clip.size(0)
        device = query_clip.device

        memory = self.encode_context(query_clip, neighbor_clips, neighbor_s2)

        generated_embeds = self.s2_embedder.bos_embed.expand(b_size, 1, -1).clone()
        predicted_tokens = torch.zeros(b_size, 21, dtype=torch.long, device=device)

        for step in range(21):
            seq_len = generated_embeds.size(1)
            decoder_input = generated_embeds + self.decoder_pos[:, :seq_len, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
            decoder_out = self.transformer.decoder(
                tgt=decoder_input, memory=memory, tgt_mask=tgt_mask
            )
            last_hidden = decoder_out[:, -1, :]  # representation for position `step`

            if step == 0:
                logits = self.head_face(last_hidden)  # [B, 6]
            else:
                logits = self.head_quadrants[step - 1](last_hidden)  # [B, 4]

            next_token = logits.argmax(dim=-1)  # [B]
            predicted_tokens[:, step] = next_token

            if step < 20:
                if step == 0:
                    next_embed = self.s2_embedder.face_embed(next_token).unsqueeze(1)
                else:
                    next_embed = self.s2_embedder.quadrant_embeds[step - 1](
                        next_token
                    ).unsqueeze(1)
                generated_embeds = torch.cat([generated_embeds, next_embed], dim=1)

        if was_training:
            self.train()

        return predicted_tokens

    @torch.no_grad()
    def generate_sample(
        self,
        query_clip: torch.Tensor,
        neighbor_clips: torch.Tensor,
        neighbor_s2: torch.Tensor,
        temperature: float = 0.7,
        num_samples: int = 10,
    ):
        """Autoregressive sampling decode — draws `num_samples` independent
        candidate sequences per query image, for building a pool to apply a
        selection strategy over (e.g. log-probability selection). Mirrors
        generate_greedy() exactly, except at each step a token is sampled
        from softmax(logits / temperature) instead of taking the argmax,
        and each candidate's cumulative log-probability under the model's
        own (untempered) distribution is tracked alongside it — this is
        the log-probability used by log-probability selection, matching
        the paper's score(s) = sum_j log p(k_j | k_1..k_{j-1}).

        Args:
            query_clip: [B, 768]
            neighbor_clips: [B, K_neighbors, 768]
            neighbor_s2: [B, K_neighbors, 21]
            temperature: softmax temperature for sampling only (<1 sharpens,
                >1 flattens); does not affect the recorded log-probability.
            num_samples: number of independent candidates per query image.

        Returns:
            predicted_tokens: LongTensor [B, num_samples, 21]
            log_probs: FloatTensor [B, num_samples] — summed log-probability
                of each candidate's 21 chosen tokens.
        """
        was_training = self.training
        self.eval()

        b_size = query_clip.size(0)
        device = query_clip.device
        k = num_samples

        # Repeat each query/neighbor context k times so all k candidates per
        # image decode together as one batched pass, instead of k sequential
        # passes.
        query_rep = query_clip.repeat_interleave(k, dim=0)
        neighbor_clips_rep = neighbor_clips.repeat_interleave(k, dim=0)
        neighbor_s2_rep = neighbor_s2.repeat_interleave(k, dim=0)

        memory = self.encode_context(query_rep, neighbor_clips_rep, neighbor_s2_rep)

        bk_size = b_size * k
        generated_embeds = self.s2_embedder.bos_embed.expand(bk_size, 1, -1).clone()
        predicted_tokens = torch.zeros(bk_size, 21, dtype=torch.long, device=device)
        log_probs = torch.zeros(bk_size, device=device)

        for step in range(21):
            seq_len = generated_embeds.size(1)
            decoder_input = generated_embeds + self.decoder_pos[:, :seq_len, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
            decoder_out = self.transformer.decoder(
                tgt=decoder_input, memory=memory, tgt_mask=tgt_mask
            )
            last_hidden = decoder_out[:, -1, :]

            if step == 0:
                logits = self.head_face(last_hidden)  # [B*K, 6]
            else:
                logits = self.head_quadrants[step - 1](last_hidden)  # [B*K, 4]

            log_p_all = torch.log_softmax(logits, dim=-1)
            sample_probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(sample_probs, num_samples=1).squeeze(-1)

            predicted_tokens[:, step] = next_token
            log_probs += log_p_all.gather(1, next_token.unsqueeze(-1)).squeeze(-1)

            if step < 20:
                if step == 0:
                    next_embed = self.s2_embedder.face_embed(next_token).unsqueeze(1)
                else:
                    next_embed = self.s2_embedder.quadrant_embeds[step - 1](
                        next_token
                    ).unsqueeze(1)
                generated_embeds = torch.cat([generated_embeds, next_embed], dim=1)

        if was_training:
            self.train()

        return predicted_tokens.view(b_size, k, 21), log_probs.view(b_size, k)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoTokenTransformer().to(device)

    q_clip = torch.randn(4, 768, device=device)
    n_clip = torch.randn(4, 15, 768, device=device)
    n_s2 = torch.randint(0, 4, (4, 15, 21), device=device)
    n_s2[:, :, 0] = torch.randint(0, 6, (4, 15), device=device)

    targets = torch.randint(0, 4, (4, 21), device=device)
    targets[:, 0] = torch.randint(0, 6, (4,), device=device)

    logits_f, logits_q = model(q_clip, n_clip, n_s2, targets)
    print("=" * 60)
    print("[TEST forward() — teacher forcing]")
    print(f"  - Face Logits Shape: {logits_f.shape} (Expected [4, 6])")
    print(f"  - Quadrant Logits Length: {len(logits_q)} x {logits_q[0].shape} (Expected 20 x [4, 4])")

    predicted = model.generate_greedy(q_clip, n_clip, n_s2)
    print("\n[TEST generate_greedy() — real inference]")
    print(f"  - Predicted tokens shape: {predicted.shape} (Expected [4, 21])")
    print(f"  - Face tokens in range [0,5]? {(predicted[:, 0] < 6).all().item()}")
    print(f"  - Quadrant tokens in range [0,3]? {(predicted[:, 1:] < 4).all().item()}")
    print("=" * 60)
