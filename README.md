# CubicSplat

Reproduction code for **CubicSplat**.

| Resource | Link |
| --- | --- |
| Paper | TBD |
| Project Page | TBD |
| Code | This repository |


## Installation

```bash
uv sync
```

## Data

Download the full benchmark datasets before running the paper script:

```bash
bash tools/download_div2k.sh
bash tools/download_kodak.sh
```

## Full Reproduction

After downloading DIV2K and Kodak, run:

```bash
bash tools/experiment.sh
```

## Evaluation

Aggregate metrics from generated outputs:

```bash
uv run python evaluate.py output/ --ref-dir datasets/DIV2K_HR --lpips --ray-mode --ray-auto-scan
uv run python evaluate.py output_kodak/ --ref-dir datasets/kodak --lpips --ray-mode --ray-auto-scan
```

## Citation

Citation metadata will be added after the paper entry is finalized.
