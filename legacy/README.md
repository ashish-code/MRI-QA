# Legacy Chainer Code

These files implement an earlier IQA pipeline using the **Chainer** deep learning framework (pre-PyTorch).
They are preserved for reference but are **not integrated** with the current PyTorch pipeline.

| File | Description |
|------|-------------|
| `nr_model.py` | No-Reference IQA model (Chainer) |
| `fr_model.py` | Full-Reference IQA model (Chainer) |
| `evaluate.py` | Single-image 2D patch evaluation |
| `evaluate_abide1.py` | Batch ABIDE-1 evaluation: NIfTI → PNG slices → score |

**Requirements:** Chainer ≥ 7, CUDA 8+. Not maintained.
