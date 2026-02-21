# LUT-NN

Paper: <https://arxiv.org/abs/2512.11843>
Reference C code: <https://github.com/izhikevich/SNN>

## Scope

This repo implements a paper-faithful LUT block:
- Forward: Eq. (1)–(3)
- Backward: Eq. (7)–(8), minimal-pair surrogate (Sec. III-A)

---

## Paper concept → code location

### 1) Anchor pairs per table (Sec. II, “Assign”)
Each table tracks $n_c$ random anchor pairs $(a_{ir}, b_{ir})$.
`lut_nn/lut_block.py` → `LUTBlock.__init__` (`anchors_a`, `anchors_b`).

### 2) Hash/index from spike-order comparisons (Eq. 1)
$$
j = H_i(x)=\operatorname{concat}(u_{i1}>0,\dots,u_{in_c}>0),\qquad
u_{ir}=x_{a_{ir}}-x_{b_{ir}}
$$
`lut_nn/lut_block.py` → `_LUTTransformFn.forward`:
- `diffs = x[:, a] - x[:, b]`
- `bits = (diffs > 0)`
- `idx = (bits * bit_powers).sum(...)`

### 3) LUT transform (Eq. 2–3)
$$
y = \sum_i S_{i,H_i(x)}
$$
`lut_nn/lut_block.py` → `_LUTTransformFn.forward`:
- `y = y + table[i, idx, :]`

### 4) Minimal comparison pair + bit flip (Sec. III-A, Fig. 7)
$$
r_i = \arg\min_r |u_{ir}|,\qquad \bar{j}=j\oplus 2^{r_i}
$$
`lut_nn/lut_block.py` → `_LUTTransformFn.forward`:
- `min_r = abs_diffs.argmin(...)`
- `flip_idx = idx ^ (1 << min_r)`

### 5) Alignment scalar for surrogate backprop (Eq. 7)
$$
g_i = \frac{\partial\mathcal{L}}{\partial y}\cdot(S_{i,\bar{j}}-S_{i,j})
$$
`lut_nn/lut_block.py` → `_LUTTransformFn.backward`:
- `g_i = (grad_output * (s_flip - s_cur)).sum(...)`

### 6) Backprop to selected latency coordinates (Eq. 8)
$$
\frac{\partial\mathcal{L}}{\partial x_{a_i}}=U'(u_i)g_i,
\qquad
\frac{\partial\mathcal{L}}{\partial x_{b_i}}=-U'(u_i)g_i
$$
`lut_nn/lut_block.py` → `_LUTTransformFn.backward`:
- `coeff = u_prime * g_i`
- `scatter_add` on `a_idx` with `+coeff`, `b_idx` with `-coeff`

### 7) Uncertainty derivative used in training (Sec. III-A)
$$
U(u)=\frac{0.5}{1+|u|},
\qquad
U'(u)=-\frac{0.5\,\operatorname{sign}(u)}{(1+|u|)^2}
$$
`lut_nn/lut_block.py` → `_LUTTransformFn.backward`:
- `u_prime = -0.5 * sign(u) / (1+|u|)^2`

### 8) Residual LUT block form (Eq. 10)
$$
x^{l+1}=x^l+S_{x^l}
$$
`lut_nn/lut_block.py` → `LUTBlock.forward`:
- `if self.residual: y = y + x`

---

## Quick MNIST sanity run

```bash
PYTHONPATH=. python3 scripts/mnist_lut_mlp.py \
  --device cuda \
  --epochs 1 \
  --train-limit 2000 \
  --test-limit 500 \
  --batch-size 64 \
  --num-tables 4 \
  --num-comparisons 4
```
