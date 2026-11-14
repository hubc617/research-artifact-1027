# Tested Environment

GPU validation on 2026-09-10 used Python 3.11.16 and PyTorch 2.1.0+cu121
with CUDA 12.1.

## Core Training

- NumPy 1.26.4
- PyTorch 2.1.0+cu121
- PyYAML 6.0.1
- scikit-learn 1.3.2

## Analysis

- pandas 2.1.4
- SciPy 1.11.4

## Visualization and Tests

- matplotlib 3.8.2
- MNE 1.6.1
- pytest 9.1.1

Install the `analysis` extra for statistics and aggregation, `analysis` and
`visualization` for Fig. 2, and `test` for pytest. The
[README install command](README.md#environment) includes all three.
