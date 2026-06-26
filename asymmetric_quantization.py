"""
Asymmetric (Affine) 8-bit Quantization
======================================

Asymmetric quantization maps an arbitrary float range [min, max] -- which need
NOT be centered on zero -- onto the unsigned integer range [0, 255]. Because the
range can be lopsided, we need a "zero-point": the integer code that represents
the real value 0.0.

    scale      = (max - min) / (qmax - qmin)
    zero_point = round(qmin - min / scale)          (then clamp)
    q          = round(x / scale) + zero_point       (then clamp)   -> quantize
    x_hat      = (q - zero_point) * scale                            -> dequantize

WHERE ASYMMETRIC QUANTIZATION IS GOOD
-------------------------------------
- One-sided / skewed data. The classic case is ReLU activations (all >= 0): the
  full [0, 255] range is spent on the values that actually occur, giving ~2x the
  resolution of symmetric quantization on the same data.
- Activations in general. Activation distributions shift with the input and are
  often not centered on zero, so the extra offset pays off.
- Any tensor whose useful range is far from being symmetric about zero.

WHERE IT IS A POOR FIT
----------------------
- Weights, and zero-centered data. The zero-point is then ~middle of the range
  and buys little accuracy while it COSTS performance: the integer matmul gains
  extra cross terms involving the zero-point, making INT8 kernels slower. For
  those, prefer symmetric quantization (see symmetric_quantization.py).
"""

import numpy as np


# Unsigned int8 range: asymmetric quant typically targets [0, 255] so the
# zero-point can sit anywhere inside the range.
QMIN = 0
QMAX = 255


def compute_scale_and_zero_point(x: np.ndarray):
    """Derive (scale, zero_point) from the observed min/max of `x`."""
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    # Always include 0.0 in the range so that real zero is representable.
    # (Important for padding, ReLU's exact zeros, etc.)
    x_min = min(x_min, 0.0)
    x_max = max(x_max, 0.0)

    if x_max == x_min:
        return 1.0, QMIN

    scale = (x_max - x_min) / (QMAX - QMIN)

    # The zero-point is where real 0.0 lands on the integer axis; clamp it so it
    # is always a valid code in [0, 255].
    zero_point = round(QMIN - x_min / scale)
    zero_point = int(np.clip(zero_point, QMIN, QMAX))

    return scale, zero_point


def quantize(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Map floats -> uint8 codes using scale and the integer offset."""
    q = np.round(x / scale) + zero_point
    q = np.clip(q, QMIN, QMAX)
    return q.astype(np.uint8)


def dequantize(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Map uint8 codes back to approximate floats."""
    return (q.astype(np.float32) - zero_point) * scale


def quantize_dequantize(x: np.ndarray):
    """Convenience: full round trip. Returns (recovered, codes, scale, zp)."""
    scale, zero_point = compute_scale_and_zero_point(x)
    q = quantize(x, scale, zero_point)
    x_hat = dequantize(q, scale, zero_point)
    return x_hat, q, scale, zero_point


if __name__ == "__main__":
    # A one-sided signal (e.g. ReLU activations): all values >= 0.
    x = np.array([0.0, 0.1, 0.4, 1.0, 2.5, 6.0], dtype=np.float32)

    x_hat, q, scale, zp = quantize_dequantize(x)

    print("Asymmetric 8-bit quantization")
    print(f"  scale (step size) : {scale:.6f}")
    print(f"  zero_point        : {zp}")
    print(f"  original  : {x}")
    print(f"  uint8 codes: {q}")
    print(f"  recovered : {np.round(x_hat, 4)}")
    print(f"  max abs error: {np.max(np.abs(x - x_hat)):.6f}")
