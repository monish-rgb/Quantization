"""
Zero-Point Quantization in a Real Operation: INT8 Matrix Multiply
=================================================================

Goal: compute  Y = X @ W  using only 8-bit integer math, then recover the
float result. This is exactly what an INT8 inference kernel does for a linear /
fully-connected layer. It shows *why* the zero-point creates extra work.

--------------------------------------------------------------------------------
THE CORE IDEA
--------------------------------------------------------------------------------
Asymmetric (affine) quantization represents a real value as:

        x  ~=  s_x * (q_x - z_x)

    s_x = scale       (float, the size of one integer step)
    q_x = code        (uint8 in [0, 255])
    z_x = zero-point  (the integer code that means real 0.0)

So a single element of the matrix product Y = X @ W is:

    Y_ij = sum_k  x_ik * w_kj
         = sum_k  [ s_x (q_x_ik - z_x) ] * [ s_w (q_w_kj - z_w) ]
         = s_x * s_w * sum_k (q_x_ik - z_x) * (q_w_kj - z_w)

Expanding the product (q_x - z_x)(q_w - z_w) gives FOUR terms:

    sum_k q_x*q_w                      <- (1) the real integer matmul (q_x @ q_w)
  - z_w * sum_k q_x                    <- (2) row sums of X, scaled by z_w
  - z_x * sum_k q_w                    <- (3) col sums of W, scaled by z_x
  + K * z_x * z_w                      <- (4) constant correction (K = inner dim)

Finally:

    Y_ij = s_x * s_w * [ term1 - term2 - term3 + term4 ]

--------------------------------------------------------------------------------
WHY THIS MATTERS  (the trade-off the comments in the other files mention)
--------------------------------------------------------------------------------
- Term (1) is the cheap, hardware-accelerated  uint8 @ uint8 -> int32  matmul.
- Terms (2), (3), (4) exist ONLY because the zero-points are non-zero. They are
  the "cross terms" that make asymmetric INT8 kernels slower than symmetric ones.
- If we had used SYMMETRIC quantization (z_x = z_w = 0), terms 2, 3, 4 all
  vanish and we are left with just  s_x * s_w * (q_x @ q_w). That is precisely
  why weights are usually quantized symmetrically.
"""

import numpy as np

import asymmetric_quantization as asym


def quantize_matrix(m: np.ndarray):
    """Quantize a whole matrix with one (scale, zero_point) -- 'per-tensor'."""
    scale, zp = asym.compute_scale_and_zero_point(m)
    q = asym.quantize(m, scale, zp)
    return q, scale, zp


def int8_matmul(X: np.ndarray, W: np.ndarray):
    """Compute X @ W using the zero-point expansion above, in integer math."""
    qX, sX, zX = quantize_matrix(X)
    qW, sW, zW = quantize_matrix(W)

    K = X.shape[1]  # the shared inner dimension (number of MACs per output)

    # Promote to int32 so the sums/products never overflow uint8.
    qX_i = qX.astype(np.int32)
    qW_i = qW.astype(np.int32)

    # (1) The actual integer matrix multiply: uint8 @ uint8 -> int32.
    term1 = qX_i @ qW_i                          # shape (M, N)

    # (2) z_w * (row-sums of qX), broadcast across the N output columns.
    term2 = zW * qX_i.sum(axis=1, keepdims=True)  # shape (M, 1)

    # (3) z_x * (col-sums of qW), broadcast across the M output rows.
    term3 = zX * qW_i.sum(axis=0, keepdims=True)  # shape (1, N)

    # (4) Constant correction added back once per output element.
    term4 = K * zX * zW                           # scalar

    acc = term1 - term2 - term3 + term4           # int32 accumulator

    # Re-apply the combined float scale to get back to real units.
    Y = (sX * sW) * acc.astype(np.float32)
    return Y, (qX, sX, zX), (qW, sW, zW)


def main():
    rng = np.random.default_rng(42)

    # ---- Dummy matrices --------------------------------------------------
    # X: activations (one-sided-ish, shifted positive) -> good fit for asymmetric.
    # W: weights (centered near zero).
    X = rng.normal(loc=1.5, scale=0.8, size=(3, 4)).astype(np.float32)
    W = rng.normal(loc=0.0, scale=0.5, size=(4, 2)).astype(np.float32)

    print("X (3x4) activations:\n", np.round(X, 3))
    print("\nW (4x2) weights:\n", np.round(W, 3))

    # ---- Ground truth: plain float matmul --------------------------------
    Y_true = X @ W

    # ---- Quantized integer matmul ----------------------------------------
    Y_q, (qX, sX, zX), (qW, sW, zW) = int8_matmul(X, W)

    print("\nQuantization parameters")
    print(f"  X: scale={sX:.5f}  zero_point={zX}")
    print(f"  W: scale={sW:.5f}  zero_point={zW}")
    print("\n  qX (uint8 codes):\n", qX)
    print("  qW (uint8 codes):\n", qW)

    print("\nResult Y = X @ W")
    print("  float (true):\n", np.round(Y_true, 4))
    print("  int8  (quant):\n", np.round(Y_q, 4))

    abs_err = np.abs(Y_true - Y_q)
    print(f"\n  max abs error : {abs_err.max():.5f}")
    print(f"  mean abs error: {abs_err.mean():.5f}")
    rel = abs_err.max() / (np.abs(Y_true).max() + 1e-12)
    print(f"  max relative error vs |Y|max: {rel:.2%}")


if __name__ == "__main__":
    main()
