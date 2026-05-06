# Neural Rule Inducer (NRI)

Zero-shot induction of interpretable DNF rules from Boolean examples.

## Description

Neural Rule Inducer (NRI) is a pretrained model for zero-shot logical rule induction. Instead of encoding literal identities, it represents literals using domain-agnostic statistical properties such as class-conditional rates, entropy, and co-occurrence. It is trained on synthetic Boolean formulas and outputs interpretable disjunctive normal form (DNF) rules for new tasks without retraining.

[Paper](TODO: link to IJCAI 2026 proceedings)

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

The package requires `torch>=2.2` and `numpy>=1.24`. A GPU is required for training; CPU execution is sufficient for small evaluation runs.

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

Evaluate one dataset:

```bash
python evaluate_uci.py \
  --checkpoint runs/quickstart/checkpoints/checkpoint_latest.pt \
  --data-dir data/uci \
  --dataset <dataset-name>
```

Evaluate all dataset subdirectories containing `X_bool.npy` and `y.npy`:

```bash
python evaluate_uci.py \
  --checkpoint runs/quickstart/checkpoints/checkpoint_latest.pt \
  --data-dir data/uci \
  --all
```

The evaluation script uses 5-fold stratified cross-validation internally. For multi-class labels, it builds one-vs-rest binary tasks and aggregates their predictions.

## Configuration

`configs/default.json` is consumed by `train.py` and contains the default data, model, optimizer, loss, and runtime settings. Main knobs include model size (`literal_embed_dim`, `clause_hidden_dim`), rule capacity (`t_max`, `k_max`), and optimizer/runtime settings (`learning_rate`, `batch_size`, `total_steps`). See paper Section 5 for the hyperparameter values used in the experiments.

## Reproducing paper results

The paper reports 69.7% mean accuracy over 14 UCI datasets (Table 1). Reproducing that number requires a trained checkpoint plus the same UCI preprocessing and 5-fold stratified cross-validation protocol used by `evaluate_uci.py`.

For this release, running the default training command with `--seed 42` for 500 steps produced a checkpoint with 90.74% synthetic validation accuracy and 75.60% mean UCI accuracy across the 14 paper datasets using `python evaluate_uci.py --checkpoint /tmp/public_ref/checkpoints/checkpoint_best.pt --data-dir <uci_data> --all`. The training command was `torchrun --nproc_per_node=1 --standalone train.py --config configs/default.json --output-dir /tmp/public_ref --total-steps 500 --batch-size 8192 --train-num-workers 4 --seed 42 --force-overwrite`, and it took 151.5 seconds, about 2.5 minutes, on one NVIDIA RTX 6000 Pro with 96 GB VRAM.

Earlier 10-seed studies on this codebase found mean UCI accuracy ranging from approximately 65% to 77.7%, with mean near 73.5% and std about 4%. The paper's 69.7% is within this distribution but on the lower end; the +5.9 pp gap between this run and the paper number is consistent with cross-seed variance plus minor codebase evolution since paper submission.

| Dataset | This run | Paper Table 1 |
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

Per-dataset accuracies are printed to stdout by `evaluate_uci.py`; to save them, redirect the output: `python evaluate_uci.py ... --all > uci_results.txt`. The trained reference checkpoint will be released as a GitHub Release artifact, not committed to the repository, because it is about 107 MB.

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
