# Image Super-Resolution Benchmark

A modular, reproducible benchmark pipeline and interactive web interface for comparing classical and deep learning image super-resolution methods across datasets and degradation conditions.

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch interactive web interface
python scripts/run_app.py --host 0.0.0.0 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to test live upscaling, split-slider comparisons, difference maps, and metric exports.

---

## 📖 Documentation

All detailed guides, specifications, presentation slides, and reports are located in the [`docs/`](docs/) directory:

| Document | Purpose |
|---|---|
| **[docs/PRESENTATION.pdf](docs/Presentation.pdf)** | Project presentation slide deck covering methodology, model design, benchmark findings, and visual comparisons. |
| **[docs/REPORT.pdf](docs/Report.pdf)** | Comprehensive final project report detailing research findings, mathematical formulations, and quantitative evaluations. |
| **[docs/SETUP.md](docs/SETUP.md)** | Environment setup, Python virtualenvs, CUDA drivers, dataset downloads, smoke tests, and Web GUI guide. |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Pipeline overview, classical interpolation baselines, deep models (SRCNN, ESPCN, EDSR-Lite), and the custom `OurNN` architecture with ablation study. |
| **[docs/TRAIN.md](docs/TRAIN.md)** | Local training, CLI overrides, remote GPU training on Kaggle via `.env`, and results aggregation. |
| **[docs/BENCHMARK.md](docs/BENCHMARK.md)** | Quantitative benchmark tables (Set5, Set14, BSD100, Urban100), ablation analysis, parameter efficiency, and qualitative upscale comparisons. |

---

## 🔬 Core Features

* **Classical Baselines**: Nearest Neighbor, Bilinear, Bicubic, and Lanczos.
* **Deep Neural Networks**: SRCNN, ESPCN, EDSR-Lite, and custom `OurNN` (incorporating fixed Sobel/Laplacian edge priors, residual blocks, Squeeze-and-Excitation channel attention, and global bicubic skip connections).
* **Degradation Pipelines**: Configurable downsampling (x2, x3, x4) with optional additive Gaussian noise (sigma 15).
* **Standardized Evaluation**: Evaluation protocol matching published literature (ITU-R BT.601 Y-channel conversion, 8-bit quantization, and boundary cropping).
* **Automated Cloud Training**: One-command Kaggle GPU remote training and sync via `scripts/push_kaggle.py`.
* **Interactive Web GUI**: FastAPI backend with side-by-side comparison cards, interactive split slider, error difference heatmaps, 100% pixel zoom, and ZIP bundle exporting.

---

## 📦 Datasets

Standard benchmark datasets used for training and quantitative evaluation:

* **Training Set**: [DIV2K_train_HR (800 images, 3.3 GB)](https://drive.google.com/file/d/172UOwD8apo8iCaBMXmr3vjeYqGn2ZIBo/view?usp=sharing)
* **Test Sets**: [Set5, Set14, BSD100, Urban100 (ZIP)](https://drive.google.com/file/d/1Uf9X0AT7WozUBIQMhgyn9tcApTVcwyWE/view?usp=sharing)

See **[docs/SETUP.md](docs/SETUP.md)** for folder placement and data verification instructions.

---

## 🧪 Quick Test (Smoke Test)

Verify your local environment and data pipeline in under a minute:

```bash
python scripts/make_dummy_data.py --out data/dummy
python scripts/train.py --config configs/smoke_test.yaml
pytest
```
