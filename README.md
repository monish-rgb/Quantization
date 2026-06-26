# 8-bit Quantization - From Scratch in NumPy

A small, heavily-commented educational project that implements **8-bit integer
quantization** of floating-point tensors using only NumPy. It covers both
quantization schemes used in modern ML inference and shows *why* each one exists
through a worked INT8 matrix-multiply example.

The goal is clarity over performance: every file is self-contained and the math
is spelled out in the docstrings so you can read the code top-to-bottom and
understand exactly what a quantizer does.

## What is quantization?

Quantization maps high-precision floating-point values onto a small set of
integer codes (here, 8-bit). It cuts memory and bandwidth by ~4x versus FP32 and
lets hardware use fast integer arithmetic, at the cost of a small, controlled
loss of precision. It is the backbone of efficient neural-network inference.

A value is reconstructed as:

```
x ≈ scale * (q - zero_point)
```

- `scale` — the real-world size of one integer step
- `q` — the integer code
- `zero_point` — the integer code that represents real `0.0` (0 for symmetric)

## Contents

| File | What it does |
| --- | --- |
| [`symmetric_quantization.py`](symmetric_quantization.py) | Symmetric (zero-point = 0) int8 quantization over `[-127, 127]`. Best for **weights** and zero-centered data. |
| [`asymmetric_quantization.py`](asymmetric_quantization.py) | Asymmetric / affine uint8 quantization over `[0, 255]` with a zero-point. Best for **activations** and one-sided/skewed data (e.g. ReLU). |
| [`example.py`](example.py) | Runs both schemes on zero-centered weights and on ReLU activations, then reports MSE to make the "which scheme wins" rule concrete. |
| [`zeropoint_matmul.py`](zeropoint_matmul.py) | Computes `Y = X @ W` entirely in INT8 math via the zero-point expansion, showing the extra "cross terms" that make asymmetric kernels slower than symmetric ones. |

## Symmetric vs. asymmetric — the rule of thumb

| | Symmetric | Asymmetric (affine) |
| --- | --- | --- |
| Integer range | `[-127, 127]` (signed) | `[0, 255]` (unsigned) |
| Zero-point | always `0` | tracked offset |
| Best for | weights, zero-centered tensors | activations, skewed/one-sided data |
| Kernel cost | cheap (`q_x @ q_w` only) | extra cross terms from non-zero zero-points |

Symmetric quantization wastes no codes on an offset and keeps INT8 GEMM kernels
simple, which is why inference engines almost always quantize **weights**
symmetrically. Asymmetric quantization spends its full range on the values that
actually occur, which is a big win for one-sided **activations** like ReLU
outputs.

## Quick start

Requires Python 3.8+ and NumPy.

```bash
pip install -r requirements.txt
```

Run any file directly to see a worked demo:

```bash
python symmetric_quantization.py    # symmetric round-trip on a weight slice
python asymmetric_quantization.py   # asymmetric round-trip on ReLU-like data
python example.py                   # side-by-side comparison + MSE
python zeropoint_matmul.py          # INT8 matrix multiply with zero-points
```

## Example output

`example.py` quantizes the same data with both schemes and reports which has the
lower error — symmetric wins on zero-centered weights, asymmetric wins on
one-sided ReLU activations, exactly as the theory predicts.

## License

Released under the [MIT License](LICENSE).
