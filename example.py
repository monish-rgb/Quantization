"""
Worked example: when to use symmetric vs. asymmetric 8-bit quantization
=======================================================================

This script runs BOTH quantizers on TWO kinds of data and reports the error,
making the rule-of-thumb concrete:

  * Symmetric  -> best for zero-centered data (e.g. neural-network weights).
  * Asymmetric -> best for one-sided / skewed data (e.g. ReLU activations).

Run:  python example.py
"""

import numpy as np

import symmetric_quantization as sym
import asymmetric_quantization as asym


def report(name: str, x: np.ndarray):
    """Quantize `x` with both schemes and print a side-by-side error summary."""
    sym_hat, _, sym_scale = sym.quantize_dequantize(x)
    asym_hat, _, asym_scale, asym_zp = asym.quantize_dequantize(x)

    # Mean squared error is a good single-number measure of quantization quality.
    sym_mse = np.mean((x - sym_hat) ** 2)
    asym_mse = np.mean((x - asym_hat) ** 2)

    print(f"\n=== {name} ===")
    print(f"  data range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  symmetric  MSE: {sym_mse:.3e}  (scale={sym_scale:.5f})")
    print(f"  asymmetric MSE: {asym_mse:.3e}  (scale={asym_scale:.5f}, "
          f"zero_point={asym_zp})")

    winner = "symmetric" if sym_mse < asym_mse else "asymmetric"
    print(f"  -> lower error with: {winner.upper()} quantization")


def main():
    rng = np.random.default_rng(0)

    # 1) Zero-centered data, like trained weights: a Gaussian around 0.
    #    Expectation: symmetric wins (no codes wasted on an offset, and INT8
    #    weight kernels prefer zero-point == 0 anyway).
    weights = rng.normal(loc=0.0, scale=0.3, size=10_000).astype(np.float32)
    report("Zero-centered weights  (Gaussian, mean=0)", weights)

    # 2) One-sided data, like ReLU activations: clamp negatives to 0.
    #    Expectation: asymmetric wins. Symmetric throws away the entire negative
    #    half of its range on data that is never negative, halving resolution.
    activations = np.maximum(0.0, rng.normal(loc=2.0, scale=1.5, size=10_000))
    activations = activations.astype(np.float32)
    report("ReLU activations       (one-sided, >= 0)", activations)

    print("\nTakeaway:")
    print("  Use SYMMETRIC for weights / zero-centered tensors.")
    print("  Use ASYMMETRIC for activations / skewed, one-sided tensors.")


if __name__ == "__main__":
    main()
