# Neural Rule Inducer (NRI)

Zero-shot induction of interpretable DNF rules from Boolean examples.

## Description

Neural Rule Inducer (NRI) is a pretrained model for zero-shot logical rule induction. Instead of encoding literal identities, it represents literals using domain-agnostic statistical properties such as class-conditional rates, entropy, and co-occurrence. It is trained on synthetic Boolean formulas and outputs interpretable disjunctive normal form (DNF) rules for new tasks without retraining.

## Reference checkpoint

The reference model is available on the [Hugging Face Hub](https://huggingface.co/phuayj/neural-rule-inducer). Use `pytorch_model.bin` with `RuleInducer.from_pretrained` for lightweight inference, or download `checkpoint_best.pt` for the full training state needed for resumption or checkpoint-based CLIs. This checkpoint was verified at 75.60% mean UCI accuracy, compared with the paper's Table 1 number of 69.7%. The two are not directly comparable: this checkpoint was evaluated under the public 5-fold CV protocol (1 fold ≈ 20% of data as support, 4 folds ≈ 80% as query, single seed), whereas the paper's protocol additionally subsamples the 4-fold training portion to 5% of itself (≈ 4% of the dataset) before rule induction and averages over 10 seeds, deliberately targeting a low-data regime.

[Paper (arXiv 2605.04916)](https://arxiv.org/abs/2605.04916)

[Model on Hugging Face](https://huggingface.co/phuayj/neural-rule-inducer)

## Repository contents

| Path | Purpose |
| --- | --- |
| `train.py` | Training entry point; supports distributed data parallel training via `torchrun`. |
| `evaluate_uci.py` | UCI dataset evaluation script. |
| `configs/default.json` | Default training configuration. |
| `rule_inducer/` | Core library: model, losses, data utilities, and evaluation helpers. |
| `pyproject.toml` | Project metadata and dependencies. |

## Installation

Run the commands below from this repository directory after cloning:

```bash
git clone https://github.com/phuayj/neural-rule-inducer.git
cd neural-rule-inducer
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The package requires `torch>=2.2` and `numpy>=1.24`; `huggingface_hub>=0.24` is also required when using `RuleInducer.from_pretrained`. A GPU is required for training; CPU execution is sufficient for small evaluation runs.

## Quickstart: training from scratch on synthetic data

From the repository directory:

```bash
torchrun --nproc_per_node=<n_gpus> --standalone train.py \
  --config configs/default.json \
  --output-dir runs/quickstart
```

Replace `<n_gpus>` with the number of GPUs on your machine. The paper used 3 GPUs.

## Quickstart: evaluating on UCI

Prepare each dataset as a directory containing `X_bool.npy` and `y.npy`:

```text
data/uci/<dataset-name>/X_bool.npy
data/uci/<dataset-name>/y.npy
```

`X_bool.npy` should have shape `[M, N]` with values in `{0, 1}` or `NaN` for missing values. `y.npy` should have shape `[M]`. Users are responsible for binarizing UCI data before evaluation; in the paper, continuous features are binarized with median thresholding (Section 5.1). A binarization script will be provided in a future release.

Choose one of two checkpoint loading paths:

**Path A (recommended): lightweight inference weights from the Hugging Face Hub.**

```python
import torch
from rule_inducer import RuleInducer

model = RuleInducer.from_pretrained("phuayj/neural-rule-inducer")
model.eval()
```

**Path B: full training-state checkpoint for resumption or direct use with `evaluate_uci.py`.**

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('phuayj/neural-rule-inducer', 'checkpoint_best.pt'))"
# then pass that path to evaluate_uci.py as before
```

Evaluate one dataset with a full checkpoint path:

```bash
python evaluate_uci.py \
  --checkpoint <path-to-checkpoint_best.pt> \
  --data-dir data/uci \
  --dataset <dataset-name>
```

Evaluate all dataset subdirectories containing `X_bool.npy` and `y.npy`:

```bash
python evaluate_uci.py \
  --checkpoint <path-to-checkpoint_best.pt> \
  --data-dir data/uci \
  --all
```

The evaluation script uses 5-fold stratified cross-validation internally. For multi-class labels, it builds one-vs-rest binary tasks and aggregates their predictions.

## Configuration

`configs/default.json` is consumed by `train.py` and contains the default data, model, optimizer, loss, and runtime settings. Main knobs include model size (`literal_embed_dim`, `clause_hidden_dim`), rule capacity (`t_max`, `k_max`), and optimizer/runtime settings (`learning_rate`, `batch_size`, `total_steps`). See paper Section 5 for the hyperparameter values used in the experiments.

## Reproducing paper results

The paper reports 69.7% mean UCI accuracy averaged across 10 seeds (Table 1). The paper's evaluation protocol applies an additional 5% stratified subsampling to the 4-fold training portion before rule induction, leaving roughly 4% of the dataset as the support set per fold; this targets the low-data regime described in paper Section 5.1. **Reproducing the paper's exact number requires the private evaluation harness with `--train-percentage 5.0`; the public `evaluate_uci.py` shipped here does not yet expose this flag and instead uses the entire 1-fold support set (≈20% of data).**

For this release, the reference checkpoint was produced with the following training command (single seed, 500 steps, batch size 8192; ≈2.5 minutes on one NVIDIA RTX 6000 Pro with 96 GB VRAM):

```bash
torchrun --nproc_per_node=1 --standalone train.py \
    --config configs/default.json \
    --output-dir runs/reference \
    --total-steps 500 \
    --batch-size 8192 \
    --train-num-workers 4 \
    --seed 42 \
    --force-overwrite
```

The resulting `runs/reference/checkpoints/checkpoint_best.pt` reaches 90.74% synthetic validation accuracy and 75.60% mean UCI accuracy across the 14 paper datasets under the public protocol:

```bash
python evaluate_uci.py \
    --checkpoint runs/reference/checkpoints/checkpoint_best.pt \
    --data-dir <path-to-uci-data> \
    --all
```

If you don't want to retrain, the same `checkpoint_best.pt` is available on the Hugging Face Hub:

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('phuayj/neural-rule-inducer', 'checkpoint_best.pt'))"
```

The +5.9 pp gap between this release reference run and the paper number is dominated by the public protocol using about 5× more support data per fold (≈20% of the dataset instead of ≈4%); minor codebase improvements since paper submission may also contribute.

These per-dataset columns use different evaluation protocols and are included as a factual reference rather than an apples-to-apples comparison.

| Dataset | This checkpoint (1 seed, ≈20% support) | Paper Table 1 (10 seeds, ≈4% support) |
| --- | --- | --- |
| adult | 65.01 | 69.6 ± 4.4 |
| breast-cancer-wisconsin | 91.42 | 88.3 ± 0.3 |
| car | 71.07 | 51.2 ± 5.4 |
| credit-approval | 85.11 | 71.5 ± 7.3 |
| diabetes | 69.50 | 68.0 ± 2.4 |
| german-credit | 69.58 | 59.8 ± 3.8 |
| hepatitis | 66.78 | 55.9 ± 3.7 |
| ionosphere | 74.93 | 62.8 ± 3.9 |
| kr-vs-kp | 67.77 | 72.3 ± 5.3 |
| mushroom | 89.40 | 87.8 ± 3.9 |
| nursery | 74.56 | 71.3 ± 4.3 |
| spambase | 70.71 | 71.9 ± 0.7 |
| tic-tac-toe | 69.94 | 56.6 ± 2.5 |
| vote | 92.59 | 88.3 ± 1.8 |
| **Mean** | **75.60** | **69.7 ± 12.0** |

Per-dataset accuracies are printed to stdout by `evaluate_uci.py`; to save them, redirect the output: `python evaluate_uci.py ... --all > uci_results.txt`. The trained reference model is available on the [Hugging Face Hub](https://huggingface.co/phuayj/neural-rule-inducer), with `pytorch_model.bin` for inference and `checkpoint_best.pt` for the full training state.

## Hardware requirements

Training requires at least 1 GPU with at least 16GB VRAM; the paper used 3 GPUs. With the default 500-step configuration, training should take on the order of a few minutes per GPU. Evaluation on UCI-scale datasets is CPU-suitable for inference, though GPU evaluation is also supported when CUDA is available.

## Project structure and model overview

NRI is a DNF-based rule learner. It encodes each literal using statistical properties, composes clauses in parallel with slot-based decoding, and evaluates candidate rules with differentiable t-norm/t-conorm logic. The resulting program can be discretized and exported as interpretable DNF rules for inspection and evaluation; see paper Sections 3--4 for the model components and training losses.

## Citation

```bibtex
TBD
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
