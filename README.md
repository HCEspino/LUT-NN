# LUT-NN

Paper: <https://arxiv.org/abs/2512.11843>

Reference C code: <https://github.com/izhikevich/SNN>

## Scope

This repo implements a paper-faithful LUT block:
- Forward: Eq. (1)–(3)
- Backward: Eq. (7)–(8), minimal-pair surrogate (Sec. III-A)

---

## Paper concept → code location

### 1) Anchor pairs per table (Sec. II, "Assign")

Each table tracks $n_c$ random anchor pairs $(a_{ir}, b_{ir})$.

`lut_nn/lut_block.py` → `LUTBlock.__init__` (`anchors_a`, `anchors_b`).

### 2) Hash/index from spike-order comparisons (Eq. 1)

$$
j = H_i(x)=\mathrm{concat}(u_{i1}>0,\dots,u_{in_c}>0),\qquad u_{ir}=x_{a_{ir}}-x_{b_{ir}}
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
r_i = \arg\min_r \lvert u_{ir} \rvert,\qquad \bar{j}=j\oplus 2^{r_i}
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
\frac{\partial\mathcal{L}}{\partial x_{a_i}}=U'(u_i)g_i, \qquad \frac{\partial\mathcal{L}}{\partial x_{b_i}}=-U'(u_i)g_i
$$

`lut_nn/lut_block.py` → `_LUTTransformFn.backward`:
- `coeff = u_prime * g_i`
- `scatter_add` on `a_idx` with `+coeff`, `b_idx` with `-coeff`

### 7) Uncertainty derivative used in training (Sec. III-A)

$$
U(u)=\frac{0.5}{1+\lvert u \rvert}, \qquad U'(u)=-\frac{0.5\,\mathrm{sign}(u)}{(1+\lvert u \rvert)^2}
$$

`lut_nn/lut_block.py` → `_LUTTransformFn.backward`:
- `u_prime = -0.5 * sign(u) / (1 + abs(u))^2`

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

## GPU troubleshooting (Jetson / cuSPARSELt)

If `import torch` fails with:

`ImportError: libcusparseLt.so.0: cannot open shared object file`

Root cause: PyTorch can be installed, but the dynamic linker cannot find cuSPARSELt at runtime.

### What this repo now does

Both training scripts (`scripts/mnist_lut_mlp.py` and `scripts/cartpole_lut_dqn.py`) now:
1. Prepend common CUDA/Jetson library paths to `LD_LIBRARY_PATH`
2. Preload `libcusparseLt.so.0` (when present) before importing `torch`

This avoids per-shell manual exports in typical local runs.

### One-liner fallback (manual)

If needed, run with:

```bash
LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH \
PYTHONPATH=. python3 scripts/cartpole_lut_dqn.py --episodes 1
```

### Verify GPU is active

`cartpole_lut_dqn.py` prints:

`device=cuda cuda_available=True`

at startup when GPU is correctly configured.
