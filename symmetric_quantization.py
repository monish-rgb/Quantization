"""
Symmetric 8-bit Quantization
============================

Symmetric quantization maps a floating-point range that is symmetric around
zero, [-max_abs, +max_abs], onto an integer range. Because the float zero maps
*exactly* to the integer zero, there is no "zero-point" offset to track.

    q = round(x / scale)        (then clamp to the integer range)
    x_hat = q * scale           (dequantize)

WHERE SYMMETRIC QUANTIZATION IS GOOD
------------------------------------
- Weights of neural networks. Trained weights are usually distributed roughly
  symmetrically around zero, so wasting no codes on an offset is efficient.
- Hardware/kernel friendliness. With zero-point == 0, the integer matrix
  multiply has no extra cross terms, so INT8 GEMM kernels are simpler and
  faster. This is why most inference engines use symmetric quantization for
  weights.
- Anything already centered on zero (e.g. tanh activations, residuals).

WHERE IT IS A POOR FIT
----------------------
- Strictly one-sided data such as ReLU outputs (all >= 0). Half of the integer
  range [-127, 0) is then never used, halving the effective resolution. Use
  asymmetric quantization there instead (see asymmetric_quantization.py).
"""

import numpy as np


# We use the "restricted" / signed range [-127, 127] (int8 is [-128, 127]).
# Dropping -128 keeps the range symmetric so that negating a value never
# overflows -- this is the convention used by most ML quantization libraries.
QMIN = -127
QMAX = 127


def compute_scale(x: np.ndarray) -> float:
    """Compute the scale factor from the maximum absolute value in `x`.

    The whole tensor is summarized by a single number: the largest magnitude.
    The scale is how much real-world range each integer step represents.
    """
    max_abs = np.max(np.abs(x))
    # Guard against an all-zero tensor (avoid divide-by-zero).
    if max_abs == 0:
        return 1.0
    return max_abs / QMAX


def quantize(x: np.ndarray, scale: float) -> np.ndarray:
    """Map floats -> int8 codes. Note there is no zero-point term."""
    q = np.round(x / scale)
    q = np.clip(q, QMIN, QMAX)
    return q.astype(np.int8)


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    """Map int8 codes back to approximate floats."""
    return q.astype(np.float32) * scale


def quantize_dequantize(x: np.ndarray):
    """Convenience: full round trip. Returns (recovered_floats, codes, scale)."""
    scale = compute_scale(x)
    q = quantize(x, scale)
    x_hat = dequantize(q, scale)
    return x_hat, q, scale


if __name__ == "__main__":
    # A small symmetric-ish signal (e.g. a slice of NN weights).
    x = np.array([-1.2, -0.5, 0.0, 0.3, 0.75, 1.0], dtype=np.float32)

    x_hat, q, scale = quantize_dequantize(x)

    print("Symmetric 8-bit quantization")
    print(f"  scale (step size) : {scale:.6f}")
    print(f"  original  : {x}")
    print(f"  int8 codes: {q}")
    print(f"  recovered : {np.round(x_hat, 4)}")
    print(f"  max abs error: {np.max(np.abs(x - x_hat)):.6f}")
