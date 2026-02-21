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
    def forward(ctx, x, table, anchors_a, anchors_b, bit_powers):
        # x: [B, I], table: [T, R, O]
        bsz, _ = x.shape
        num_tables, _, out_features = table.shape
        num_comp = anchors_a.shape[1]

        y = x.new_zeros(bsz, out_features)

        # Cached tensors for backward
        idx_all = torch.empty((num_tables, bsz), dtype=torch.long, device=x.device)
        flip_idx_all = torch.empty((num_tables, bsz), dtype=torch.long, device=x.device)
        min_u_all = torch.empty((num_tables, bsz), dtype=x.dtype, device=x.device)
        min_a_all = torch.empty((num_tables, bsz), dtype=torch.long, device=x.device)
        min_b_all = torch.empty((num_tables, bsz), dtype=torch.long, device=x.device)

        for i in range(num_tables):
            a = anchors_a[i]  # [C]
            b = anchors_b[i]  # [C]
            diffs = x[:, a] - x[:, b]  # [B, C]

            bits = (diffs > 0).long()  # [B, C]
            idx = (bits * bit_powers).sum(dim=1)  # [B]
            y = y + table[i, idx, :]  # Eq. (3)

            abs_diffs = diffs.abs()
            min_r = abs_diffs.argmin(dim=1)  # [B]
            min_u = diffs.gather(1, min_r.unsqueeze(1)).squeeze(1)  # [B]

            flip_bit = (1 << min_r).long()  # [B]
            flip_idx = idx ^ flip_bit

            min_a = a[min_r]  # [B]
            min_b = b[min_r]  # [B]

            idx_all[i] = idx
            flip_idx_all[i] = flip_idx
            min_u_all[i] = min_u
            min_a_all[i] = min_a
            min_b_all[i] = min_b

        ctx.save_for_backward(x, table, idx_all, flip_idx_all, min_u_all, min_a_all, min_b_all)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, table, idx_all, flip_idx_all, min_u_all, min_a_all, min_b_all = ctx.saved_tensors

        bsz, in_features = x.shape
        num_tables = table.shape[0]

        grad_x = torch.zeros_like(x)
        grad_table = torch.zeros_like(table)

        # U'(u) for U(u)=0.5/(1+|u|)
        # d/du [0.5/(1+|u|)] = -0.5*sign(u)/(1+|u|)^2
        u_prime = -0.5 * torch.sign(min_u_all) / (1.0 + min_u_all.abs()).pow(2)  # [T, B]

        for i in range(num_tables):
            idx = idx_all[i]          # [B]
            flip_idx = flip_idx_all[i]  # [B]

            s_cur = table[i, idx, :]       # [B, O]
            s_flip = table[i, flip_idx, :] # [B, O]

            # Eq. (7): g_i = dL/dy · (S_flip - S_cur)
            g_i = (grad_output * (s_flip - s_cur)).sum(dim=1)  # [B]

            coeff = u_prime[i] * g_i  # [B]

            a_idx = min_a_all[i]
            b_idx = min_b_all[i]

            # Eq. (8): +/- U'(u_i) * g_i on the minimal pair coordinates
            grad_x.scatter_add_(1, a_idx.unsqueeze(1), coeff.unsqueeze(1))
            grad_x.scatter_add_(1, b_idx.unsqueeze(1), (-coeff).unsqueeze(1))

            # Paper pseudocode line 29: update selected row by v^{l+1}
            grad_table[i].index_add_(0, idx, grad_output)

        return grad_x, grad_table, None, None, None


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
        )

        if self.residual:
            y = y + x
        return y
