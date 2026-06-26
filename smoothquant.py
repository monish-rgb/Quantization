"""
SmoothQuant: Post-Training Quantization for LLMs (W8A8)
======================================================

SmoothQuant (Xiao et al., 2022 -- https://arxiv.org/abs/2211.10438) is a
*post-training* quantization (PTQ) method: no retraining, no gradients. You take
an already-trained model, run a little calibration data through it, and emit an
INT8 model. The trick it solves is specific to large language models.

--------------------------------------------------------------------------------
THE PROBLEM: ACTIVATION OUTLIERS
--------------------------------------------------------------------------------
For a linear layer we compute  Y = X @ W, where

    X = activations,  shape (tokens, C_in)
    W = weights,      shape (C_in, C_out)

To run this in INT8 we must quantize BOTH X and W. Two empirical facts about
transformers:

  1. WEIGHTS are easy to quantize. Their values are flat and well-behaved, so a
     single per-tensor scale captures them with little error.

  2. ACTIVATIONS are hard to quantize. A few "outlier" feature CHANNELS (columns
     of X) have magnitudes 10-100x larger than all the others, and they appear
     in the SAME channels for every token. With per-tensor quantization the
     scale is dragged up by those few outliers, so every ordinary value collapses
     onto just a handful of integer codes -> huge error.

Per-channel activation quantization would fix this, but you CAN'T have a separate
scale per activation channel and still use a fast INT8 matmul (the scale would
sit on the inner/contraction dimension and couldn't be factored out). So
activations are stuck with coarse per-tensor (or per-token) scales.

--------------------------------------------------------------------------------
THE IDEA: MIGRATE DIFFICULTY FROM ACTIVATIONS TO WEIGHTS
--------------------------------------------------------------------------------
Activations are hard, weights are easy -- so move some of the "hardness" from X
into W, where there is room to spare. We do this with a per-input-channel
smoothing factor s (a vector of length C_in), dividing it out of X and
multiplying it into W:

        X_smooth = X / s            (shrinks the outlier channels of X)
        W_smooth = diag(s) @ W      (grows the matching rows of W)

This is MATHEMATICALLY EQUIVALENT to the original layer -- it changes nothing in
full precision:

        X_smooth @ W_smooth = (X / s) @ (diag(s) @ W)
                            = X @ diag(1/s) @ diag(s) @ W
                            = X @ W = Y

s is folded offline: into W permanently, and into the *previous* layer's
output (e.g. LayerNorm scale) so X arrives pre-divided at runtime. There is no
extra runtime cost. Only AFTER smoothing do we quantize, and now both tensors
are easy.

--------------------------------------------------------------------------------
CHOOSING s: THE MIGRATION STRENGTH alpha
--------------------------------------------------------------------------------
For each input channel j we balance the per-channel maxima of X and W:

        s_j = max(|X_:,j|)^alpha  /  max(|W_j,:|)^(1 - alpha)

  - alpha = 0 : s = 1, do nothing (original model).
  - alpha = 1 : push ALL the difficulty onto the weights.
  - alpha = 0.5 : split it evenly. A good default for most LLMs; models with very
    severe outliers (e.g. some 6.7B+ models) prefer ~0.75.

Intuition: a big activation channel (large max|X|) gets a large s, so X/s shrinks
it while diag(s)@W grows the corresponding weight row by the same factor. After
smoothing, the two tensors share the difficulty instead of activations bearing
all of it.

This file shows the whole pipeline in NumPy and compares, on a layer with
planted activation outliers:

        (A) naive   W8A8 : quantize X and W directly        -> large error
        (B) SmoothQuant  : smooth first, then quantize       -> small error
"""

import numpy as np

import symmetric_quantization as sym


def per_channel_absmax(m: np.ndarray, axis: int) -> np.ndarray:
    """Largest magnitude along `axis`, i.e. one number per channel.

    For activations X (tokens, C_in) we reduce over tokens (axis=0) to get the
    max magnitude SEEN in each input channel. For weights W (C_in, C_out) we
    reduce over the output dim (axis=1) to get the max magnitude in each input
    channel's row. Both yield a vector of length C_in -- they line up.
    """
    return np.max(np.abs(m), axis=axis)


def compute_smoothing_factor(X: np.ndarray, W: np.ndarray,
                             alpha: float = 0.5) -> np.ndarray:
    """The per-input-channel scale s that balances activation vs weight ranges.

        s_j = max(|X_:,j|)^alpha / max(|W_j,:|)^(1 - alpha)

    Returns a vector of length C_in (one factor per input channel).
    """
    act_absmax = per_channel_absmax(X, axis=0)   # (C_in,) over tokens
    wgt_absmax = per_channel_absmax(W, axis=1)    # (C_in,) over output dim

    # Guard against zero channels (a dead channel would give 0**alpha = 0 and
    # then a divide-by-zero); clamp the maxima to a tiny floor.
    eps = 1e-8
    act_absmax = np.maximum(act_absmax, eps)
    wgt_absmax = np.maximum(wgt_absmax, eps)

    s = act_absmax ** alpha / wgt_absmax ** (1.0 - alpha)
    return s.astype(np.float32)


def apply_smoothing(X: np.ndarray, W: np.ndarray, s: np.ndarray):
    """Fold s out of activations and into weights. Equivalent in full precision.

        X_smooth = X / s          (s broadcasts across the C_in columns of X)
        W_smooth = s[:, None] * W (s scales each input-channel ROW of W)
    """
    X_smooth = X / s                  # (tokens, C_in) / (C_in,)  -> per column
    W_smooth = s[:, None] * W         # (C_in, 1) * (C_in, C_out) -> per row
    return X_smooth, W_smooth


def quantize_per_tensor(m: np.ndarray):
    """Symmetric per-tensor INT8 round-trip. Returns recovered floats + scale."""
    scale = sym.compute_scale(m)
    q = sym.quantize(m, scale)
    return sym.dequantize(q, scale), scale


def linear_w8a8(X: np.ndarray, W: np.ndarray):
    """Naive INT8 linear: quantize X and W directly (no smoothing), then matmul."""
    X_hat, sX = quantize_per_tensor(X)
    W_hat, sW = quantize_per_tensor(W)
    return X_hat @ W_hat, sX, sW


def linear_smoothquant(X: np.ndarray, W: np.ndarray, alpha: float = 0.5):
    """SmoothQuant INT8 linear: smooth first, THEN quantize, then matmul."""
    s = compute_smoothing_factor(X, W, alpha)
    X_smooth, W_smooth = apply_smoothing(X, W, s)

    X_hat, sX = quantize_per_tensor(X_smooth)
    W_hat, sW = quantize_per_tensor(W_smooth)
    return X_hat @ W_hat, s, sX, sW


def _summary(name: str, Y_true: np.ndarray, Y_q: np.ndarray):
    err = np.abs(Y_true - Y_q)
    rel = err.max() / (np.abs(Y_true).max() + 1e-12)
    print(f"  {name:<28} max abs err {err.max():9.4f}   "
          f"mean abs err {err.mean():8.4f}   rel {rel:6.2%}")


def main():
    rng = np.random.default_rng(0)

    tokens, C_in, C_out = 64, 32, 16

    # ---- Build a layer WITH activation outliers --------------------------
    # Ordinary activations: small, well-behaved.
    X = rng.normal(0.0, 1.0, size=(tokens, C_in)).astype(np.float32)
    # Plant a handful of OUTLIER CHANNELS: a few columns that are ~50x larger
    # and present for every token -- exactly the transformer pathology.
    outlier_channels = [3, 11, 20]
    X[:, outlier_channels] *= 50.0

    # Weights: flat and well-behaved (easy to quantize), as in real models.
    W = rng.normal(0.0, 0.5, size=(C_in, C_out)).astype(np.float32)

    # ---- Ground truth -----------------------------------------------------
    Y_true = X @ W

    # ---- (A) Naive W8A8 vs (B) SmoothQuant --------------------------------
    Y_naive, sX_n, sW_n = linear_w8a8(X, W)
    Y_smooth, s, sX_s, sW_s = linear_smoothquant(X, W, alpha=0.5)

    print("SmoothQuant demo: linear layer with planted activation outliers")
    print(f"  shape: X{X.shape} @ W{W.shape} -> Y{Y_true.shape}")
    print(f"  outlier channels: {outlier_channels} (each ~50x normal)\n")

    print("Per-tensor activation scale (smaller = finer resolution):")
    print(f"  naive       sX = {sX_n:.5f}   <- dragged up by the outliers")
    print(f"  smoothquant sX = {sX_s:.5f}   <- outliers tamed\n")

    print("Reconstruction error of Y = X @ W:")
    _summary("(A) naive W8A8", Y_true, Y_naive)
    _summary("(B) SmoothQuant (alpha=0.5)", Y_true, Y_smooth)

    # ---- alpha sweep: how migration strength trades off -------------------
    print("\nMigration-strength sweep (alpha): 0=do nothing, 1=all onto weights")
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        Yq, *_ = linear_smoothquant(X, W, alpha=alpha)
        err = np.abs(Y_true - Yq)
        print(f"  alpha={alpha:<4}  max abs err {err.max():9.4f}   "
              f"mean abs err {err.mean():8.4f}")

    print("\nTakeaway:")
    print("  Activation outliers wreck naive per-tensor INT8. SmoothQuant divides")
    print("  them out of X and folds the same factor into W (mathematically")
    print("  identical in float), so both tensors quantize cleanly. alpha controls")
    print("  how much difficulty moves from activations to weights; ~0.5 is a")
    print("  good default.")


if __name__ == "__main__":
    main()
