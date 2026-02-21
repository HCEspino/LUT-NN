import torch
import torch.nn as nn


class LUTBlock(nn.Module):
    """
    Configurable lookup-table block inspired by spike-order hashing.

    Forward idea:
    - Build binary index from pairwise anchor comparisons.
    - Fetch one row per table (hard route).
    - Also compute differentiable soft route over all rows.
    - Use straight-through trick so forward is hard, backward uses soft path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_tables: int = 8,
        num_comparisons: int = 6,
        tau: float = 0.5,
        routing_sharpness: float = 12.0,
        residual: bool = False,
        table_init_std: float = 0.02,
        seed: int | None = None,
    ):
        super().__init__()
        if num_comparisons <= 0:
            raise ValueError("num_comparisons must be > 0")
        if in_features < 2:
            raise ValueError("in_features must be >= 2")

        self.in_features = in_features
        self.out_features = out_features
        self.num_tables = num_tables
        self.num_comparisons = num_comparisons
        self.num_rows = 2 ** num_comparisons
        self.tau = tau
        self.routing_sharpness = routing_sharpness
        self.residual = residual and (in_features == out_features)

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        # Each table has num_comparisons anchor pairs (a,b).
        anchors_a = torch.randint(0, in_features, (num_tables, num_comparisons), generator=g)
        anchors_b = torch.randint(0, in_features, (num_tables, num_comparisons), generator=g)

        # Prevent trivial equal pairs when possible.
        mask_eq = anchors_a == anchors_b
        while mask_eq.any():
            anchors_b[mask_eq] = torch.randint(0, in_features, (mask_eq.sum(),), generator=g)
            mask_eq = anchors_a == anchors_b

        self.register_buffer("anchors_a", anchors_a, persistent=True)
        self.register_buffer("anchors_b", anchors_b, persistent=True)

        # LUT parameters: [tables, rows, out_features]
        table = torch.empty(num_tables, self.num_rows, out_features)
        nn.init.normal_(table, std=table_init_std)
        self.table = nn.Parameter(table)

        # Binary codebook for soft routing: [rows, num_comparisons]
        row_ids = torch.arange(self.num_rows)
        bit_positions = torch.arange(num_comparisons)
        bits = ((row_ids[:, None] >> bit_positions[None, :]) & 1).float()
        self.register_buffer("codebook_bits", bits, persistent=False)

        # For hard index build
        powers = (2 ** torch.arange(num_comparisons)).long()
        self.register_buffer("bit_powers", powers, persistent=False)

    def _pairwise_diffs(self, x: torch.Tensor, table_idx: int) -> torch.Tensor:
        a = self.anchors_a[table_idx]
        b = self.anchors_b[table_idx]
        return x[:, a] - x[:, b]  # [B, num_comparisons]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"Expected [B, {self.in_features}], got {tuple(x.shape)}")

        batch = x.size(0)
        hard_sum = x.new_zeros(batch, self.out_features)
        soft_sum = x.new_zeros(batch, self.out_features)

        for i in range(self.num_tables):
            diffs = self._pairwise_diffs(x, i)  # [B, C]

            # Hard bits for discrete routing.
            hard_bits = (diffs > 0).long()
            hard_idx = (hard_bits * self.bit_powers).sum(dim=1)  # [B]
            hard_rows = self.table[i][hard_idx]  # [B, O]

            # Soft routing distribution over all rows.
            probs = torch.sigmoid(diffs / self.tau)  # [B, C]
            dist2 = (probs[:, None, :] - self.codebook_bits[None, :, :]).pow(2).sum(dim=-1)
            logits = -self.routing_sharpness * dist2
            w = torch.softmax(logits, dim=-1)  # [B, rows]
            soft_rows = w @ self.table[i]      # [B, O]

            hard_sum = hard_sum + hard_rows
            soft_sum = soft_sum + soft_rows

        # Straight-through: hard forward, soft backward
        out = hard_sum + (soft_sum - soft_sum.detach())

        if self.residual:
            out = out + x
        return out
