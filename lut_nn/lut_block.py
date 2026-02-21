import torch
import torch.nn as nn


class _LUTTransformFn(torch.autograd.Function):
    """
    Paper-faithful LUT transform with surrogate-gradient backward.

    Forward follows Eq. (1)-(3):
      j = H_i(x) from pairwise sign comparisons
      y = sum_i S_{i, H_i(x)}

    Backward follows Eq. (7)-(8) with the paper's minimal-pair rule:
      g_i = dL/dy · (S_{i, j_flip} - S_{i, j})
      dL/dx_a += U'(u_i) * g_i
      dL/dx_b -= U'(u_i) * g_i

    where U(u)=0.5/(1+|u|), so U'(u)=-0.5*sign(u)/(1+|u|)^2.
    """

    @staticmethod
    def forward(ctx, x, table, anchors_a, anchors_b, bit_powers, surrogate_topk):
        # x: [B, I], table: [T, R, O]
        bsz, _ = x.shape
        num_tables, _, out_features = table.shape
        num_comp = anchors_a.shape[1]
        k = int(surrogate_topk)
        if k < 1:
            raise ValueError("surrogate_topk must be >= 1")
        k = min(k, num_comp)

        y = x.new_zeros(bsz, out_features)

        # Cached tensors for backward
        idx_all = torch.empty((num_tables, bsz), dtype=torch.long, device=x.device)
        topk_flip_idx_all = torch.empty((num_tables, bsz, k), dtype=torch.long, device=x.device)
        topk_u_all = torch.empty((num_tables, bsz, k), dtype=x.dtype, device=x.device)
        topk_abs_all = torch.empty((num_tables, bsz, k), dtype=x.dtype, device=x.device)
        topk_a_all = torch.empty((num_tables, bsz, k), dtype=torch.long, device=x.device)
        topk_b_all = torch.empty((num_tables, bsz, k), dtype=torch.long, device=x.device)

        for i in range(num_tables):
            a = anchors_a[i]  # [C]
            b = anchors_b[i]  # [C]
            diffs = x[:, a] - x[:, b]  # [B, C]

            bits = (diffs > 0).long()  # [B, C]
            idx = (bits * bit_powers).sum(dim=1)  # [B]
            y = y + table[i, idx, :]  # Eq. (3)

            abs_diffs = diffs.abs()
            topk_r = abs_diffs.topk(k, dim=1, largest=False).indices  # [B, K]
            topk_u = diffs.gather(1, topk_r)  # [B, K]
            topk_abs = abs_diffs.gather(1, topk_r)  # [B, K]

            flip_bit = (1 << topk_r).long()  # [B, K]
            topk_flip_idx = idx.unsqueeze(1) ^ flip_bit  # [B, K]

            topk_a = a[topk_r]  # [B, K]
            topk_b = b[topk_r]  # [B, K]

            idx_all[i] = idx
            topk_flip_idx_all[i] = topk_flip_idx
            topk_u_all[i] = topk_u
            topk_abs_all[i] = topk_abs
            topk_a_all[i] = topk_a
            topk_b_all[i] = topk_b

        ctx.save_for_backward(x, table, idx_all, topk_flip_idx_all, topk_u_all, topk_abs_all, topk_a_all, topk_b_all)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, table, idx_all, topk_flip_idx_all, topk_u_all, topk_abs_all, topk_a_all, topk_b_all = ctx.saved_tensors

        bsz, in_features = x.shape
        num_tables = table.shape[0]
        k = topk_flip_idx_all.shape[2]

        grad_x = torch.zeros_like(x)
        grad_table = torch.zeros_like(table)

        # U'(u) for U(u)=0.5/(1+|u|)
        # d/du [0.5/(1+|u|)] = -0.5*sign(u)/(1+|u|)^2
        u_prime = -0.5 * torch.sign(topk_u_all) / (1.0 + topk_u_all.abs()).pow(2)  # [T, B, K]

        for i in range(num_tables):
            idx = idx_all[i]  # [B]
            s_cur = table[i, idx, :]  # [B, O]

            flip_idx = topk_flip_idx_all[i]  # [B, K]
            s_flip = table[i, flip_idx, :]  # [B, K, O]

            # Eq. (7) generalized over top-k flips per table.
            # g_i[b, k] = dL/dy[b] · (S_flip[b, k] - S_cur[b])
            g_i = (grad_output.unsqueeze(1) * (s_flip - s_cur.unsqueeze(1))).sum(dim=2)  # [B, K]

            # Proximity weighting: closer-to-boundary pairs get more signal.
            weight = 1.0 / (1.0 + topk_abs_all[i])  # [B, K]
            weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            coeff = u_prime[i] * g_i * weight  # [B, K]

            a_idx = topk_a_all[i]  # [B, K]
            b_idx = topk_b_all[i]  # [B, K]

            # Flatten K into batch dimension for efficient indexed accumulation.
            coeff_flat = coeff.reshape(-1)  # [B*K]
            a_flat = a_idx.reshape(-1)  # [B*K]
            b_flat = b_idx.reshape(-1)  # [B*K]
            batch_idx = torch.arange(bsz, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)

            grad_x.index_put_((batch_idx, a_flat), coeff_flat, accumulate=True)
            grad_x.index_put_((batch_idx, b_flat), -coeff_flat, accumulate=True)

            # Paper pseudocode line 29: update selected row by v^{l+1}
            grad_table[i].index_add_(0, idx, grad_output)

        return grad_x, grad_table, None, None, None, None


class LUTBlock(nn.Module):
    """
    Configurable LUT block faithful to the Spiking Manifesto equations.

    Notes:
    - Forward: Eq. (1)-(3).
    - Backward: Eq. (7)-(8), minimal-anchor-pair surrogate.
    - No softmax/ST routing (those are outside the paper's core LUT equations).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_tables: int = 8,
        num_comparisons: int = 6,
        residual: bool = False,
        table_init_std: float = 0.02,
        seed: int | None = None,
        surrogate_topk: int = 1,
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
        self.residual = residual and (in_features == out_features)
        self.surrogate_topk = max(1, min(int(surrogate_topk), num_comparisons))

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        # Anchor pairs (a_ir, b_ir) per table i
        anchors_a = torch.randint(0, in_features, (num_tables, num_comparisons), generator=g)
        anchors_b = torch.randint(0, in_features, (num_tables, num_comparisons), generator=g)

        # Enforce a_ir != b_ir
        mask_eq = anchors_a == anchors_b
        while mask_eq.any():
            anchors_b[mask_eq] = torch.randint(0, in_features, (mask_eq.sum(),), generator=g)
            mask_eq = anchors_a == anchors_b

        self.register_buffer("anchors_a", anchors_a, persistent=True)
        self.register_buffer("anchors_b", anchors_b, persistent=True)

        # LUT tensor S: [tables, rows, out_features]
        table = torch.empty(num_tables, self.num_rows, out_features)
        nn.init.normal_(table, std=table_init_std)
        self.table = nn.Parameter(table)

        # Bit weights for index construction in Eq. (1)
        bit_powers = (2 ** torch.arange(num_comparisons)).long()
        self.register_buffer("bit_powers", bit_powers, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"Expected [B, {self.in_features}], got {tuple(x.shape)}")

        y = _LUTTransformFn.apply(
            x,
            self.table,
            self.anchors_a,
            self.anchors_b,
            self.bit_powers,
            self.surrogate_topk,
        )

        if self.residual:
            y = y + x
        return y
