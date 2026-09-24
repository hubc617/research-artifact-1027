# Tested Environment

This validated reference implementation was tested on 2026-09-10 with Python 3.11.16, PyTorch 2.1.0+cu121, and a CUDA 12.1 PyTorch build. The environment had CUDA available during the recorded GPU validation. No private filesystem paths are required by the released commands.

## Core Training

- NumPy 1.26.4
- PyTorch 2.1.0+cu121
- PyYAML 6.0.1
- scikit-learn 1.3.2

## Analysis

- pandas 2.1.4
- SciPy 1.11.4

`pandas` is an analysis dependency because it is imported only by the paper-result aggregation module. Core Stage 1--3 training does not import it.

## Visualization and Tests

- matplotlib 3.8.2
- MNE 1.6.1
- pytest 9.1.1

Install only the required extras: `pip install -e '.[analysis]'`, `pip install -e '.[visualization]'`, or `pip install -e '.[test]'`.
