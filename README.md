# [ECCV 2026 Oral] CubicSplat: Differentiable Vector Graphics via Error-Bounded Forward Relaxation

Official PyTorch/TileLang implementation of **CubicSplat: Differentiable Vector
Graphics via Error-Bounded Forward Relaxation**.

![CubicSplat qualitative results](https://cubicsplat.github.io/assets/showcase.webp)

<p align="center">
  <a href="https://arxiv.org/abs/2608.20803"><img src="https://img.shields.io/badge/ECCV%202026-Oral-4b44ce.svg" alt="ECCV 2026 Oral"></a>
  <a href="https://arxiv.org/abs/2608.20803"><img src="https://img.shields.io/badge/arXiv-2608.20803-b31b1b.svg" alt="arXiv"></a>
  <a href="https://cubicsplat.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://github.com/CubicSplat/repo"><img src="https://img.shields.io/badge/Code-GitHub-181717.svg?logo=github" alt="GitHub Code"></a>
</p>

| Resource | Link |
| --- | --- |
| Paper | [arXiv](https://arxiv.org/abs/2608.20803) |
| Project Page | [cubicsplat.github.io](https://cubicsplat.github.io/) |
| Code | [GitHub](https://github.com/CubicSplat/repo) |

## Abstract

Vector graphics are prized for their resolution independence, compact storage,
and direct editability, making differentiable optimization of their parametric
primitives an attractive goal. Yet classical rasterization is discontinuous with
respect to geometry, and existing remedies that smooth the forward pass demand
increasingly elaborate heuristics as scene complexity grows. We trace this
fragility to a **gradient seesaw**: design choices that improve forward
geometric exactness can systematically degrade the induced gradient signal, and
vice versa. To navigate this tension we introduce **CubicSplat**, a
differentiable vector rasterizer that replaces Bezier closest-point solvers with
uniform polyline surrogates whose geometric error is bounded at
$\mathcal{O}(S^{-2})$. The resulting static computation graph yields
well-conditioned gradients by construction, while a compositing-derived
visibility mechanism prunes degenerate primitives without auxiliary
regularization. On DIV2K and Kodak benchmarks CubicSplat achieves
state-of-the-art reconstruction quality with over **2 dB** PSNR gain in the
closed-fill setting, while training up to **4x** faster than prior methods.

## Authors

Chenglong Liu, Xin Zhang, Yimeng Zhu, Liyang He, Yixiao Ma, Yu Su, Zhenya
Huang, and Qi Liu.

State Key Laboratory of Cognitive Intelligence, University of Science and
Technology of China; Hefei Normal University; Zhejiang Key Laboratory of
Intelligent Education Technology and Application, Zhejiang Normal University.


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

```bibtex
@inproceedings{liu2026cubicsplat,
  title     = {CubicSplat: Differentiable Vector Graphics via Error-Bounded Forward Relaxation},
  author    = {Liu, Chenglong and Zhang, Xin and Zhu, Yimeng and He, Liyang and
               Ma, Yixiao and Su, Yu and Huang, Zhenya and Liu, Qi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
