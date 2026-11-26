# Reproduction environments

ARNLE-IAV uses separate environments for ELMo representation export, TensorFlow Bi-LSTM modelling, and downstream analysis. The tables below record the core package versions observed on the author server on 25 November 2026. They are an evidence record for reproduction, not a claim that every listed package is required by every script and not a complete export of all transitive packages.

## Stage-to-environment mapping

| Workflow stage | Environment | Python | Intended use |
| --- | --- | --- | --- |
| ELMo representation export | `elmo2` | 3.6.13 | Run `model/embedding/elmo_embedding_stream.py` with the pretrained ELMo model. |
| Bi-LSTM training and prediction | `arnle` | 3.6.13 | Run the TensorFlow classifier scripts and original ARNLE analysis utilities. `logomaker==0.8` is installed here. |
| Preprocessing and downstream analysis | `base` | 3.12.2 | Run sequence preprocessing, PCA, transition-band, candidate-site, robustness, plotting, and HA-NA paired analyses. |

Do not treat these three environments as interchangeable. The model environments retain the Python 3.6-era dependency stack, whereas several downstream scripts use `from __future__ import annotations` and therefore require Python 3.7 or newer.

On the author server, activate the environment corresponding to the current stage:

```bash
conda activate elmo2  # ELMo export
conda activate arnle  # TensorFlow Bi-LSTM and original ARNLE utilities
conda activate base   # preprocessing and downstream analysis
```

The root `requirements.txt` is retained as a compact dependency inventory. It should not be installed wholesale into any one of these environments.

## `elmo2`: ELMo export

Core versions observed:

| Package | Version |
| --- | --- |
| Python | 3.6.13 |
| PyTorch | 1.10.0 (`py3.6_cuda11.3_cudnn8.2.0_0`) |
| torchvision | 0.11.0 (`py36_cu113`) |
| CUDA Toolkit | 11.3.1 |
| ELMoForManyLangs | 0.0.4.post2 |
| Gensim | 4.0.1 |
| h5py | 2.10.0 |
| NumPy | 1.19.5 |
| SciPy | 1.5.2 |
| overrides | 3.1.0 |
| tqdm | 4.63.0 |
| protobuf | 3.17.2 |

The bundled ELMoForManyLangs `setup.py` in the public ARNLE repository declares `torch`, `h5py`, `numpy`, and `overrides` as installation requirements. The server environment satisfies those core imports. A package consistency check reports that `tqdm` expects `importlib-resources`, which is not installed; therefore a clean recreation should add a Python-3.6-compatible `importlib-resources` release and verify the embedding command with a small input before a full export.

## `arnle`: TensorFlow classifier and original ARNLE utilities

Core versions observed:

| Package | Version |
| --- | --- |
| Python | 3.6.13 |
| tensorflow-gpu | 2.6.0 |
| keras | 2.6.0 |
| CUDA Toolkit | 11.2.0 |
| cuDNN | 8.1.0.77 |
| NumPy | 1.19.3 |
| pandas | 0.23.0 |
| scikit-learn | 0.23.2 |
| SciPy | 1.5.4 |
| Matplotlib | 2.2.2 |
| seaborn | 0.11.2 |
| Gensim | 3.8.3 |
| h5py | 3.1.0 |
| protobuf | 3.19.6 |
| tqdm | 4.62.3 |
| logomaker | 0.8 |

`logomaker==0.8` is confirmed both in this server environment and in the public ARNLE README and `requirement.txt`. It is not part of the server `base` environment.

ELMoForManyLangs is also installed in `arnle`, but the package consistency check reports missing `torch` and `overrides`. ELMo export should therefore be run in `elmo2`; installing or changing packages in `arnle` is not required for the documented stage split.

## `base`: preprocessing and downstream analysis

Core versions observed:

| Package | Version |
| --- | --- |
| Python | 3.12.2 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| scikit-learn | 1.5.1 |
| SciPy | 1.12.0 |
| statsmodels | 0.14.2 |
| Biopython | 1.86 |
| Matplotlib | 3.9.2 |
| seaborn | 0.13.2 |
| Plotly | 5.24.1 |
| openpyxl | 3.1.5 |
| Gensim | 4.3.3 |
| h5py | 3.11.0 |
| tqdm | 4.66.5 |
| protobuf | 4.25.3 |

The package consistency check for `base` completed without reported conflicts. TensorFlow, PyTorch, ELMoForManyLangs, and logomaker are not installed in this environment, so model and sequence-logo steps must not be assigned to `base` without creating and validating a separate environment.

## Relationship to the public ARNLE declaration

The public [ARNLE README](https://github.com/SN-1604/ARNLE/blob/main/README.md) declares Python 3.6, TensorFlow 2.6.0, CUDA 11.8, cuDNN 8.9.1, NumPy 1.19.3, pandas 0.23.0, scikit-learn 0.23.2, Gensim 4.1.2, and Logomaker 0.8. Its [requirement.txt](https://github.com/SN-1604/ARNLE/blob/main/requirement.txt) lists the Python packages but not the complete ELMo/PyTorch stack.

The author-server `arnle` environment differs from that declaration in at least CUDA Toolkit (11.2.0 rather than 11.8), cuDNN (8.1.0.77 rather than 8.9.1), and Gensim (3.8.3 rather than 4.1.2). These sources are recorded separately because the repository declaration and the environment actually present on the server answer different questions. For reproduction on the author server, use the observed stage-specific environments; for a clean external reconstruction, begin with the recorded versions, resolve platform-specific CUDA packages explicitly, and validate each stage on a small input before running the full dataset.
