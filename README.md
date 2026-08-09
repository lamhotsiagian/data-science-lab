# 📘 Master Data Science Lab

Complete hands-on lab implementations for the **Master Data Science Curriculum** — 42 labs across 14 chapters covering Python foundations through production MLOps.

## 🏗️ Project Structure

```
data-science-lab/
├── ch01_programming_foundations/    # Python, Pandas, Polars, ETL pipelines
├── ch02_mathematics/                # Linear algebra, SVD, gradient descent
├── ch03_probability_statistics/     # Monte Carlo, Bayesian inference, A/B testing
├── ch04_feature_engineering/        # Feature engineering, time series, NLP preprocessing
├── ch05_visualization/              # Dashboards, BI reports, data storytelling
├── ch06_linear_models/              # OLS, regularization, diagnostics
├── ch07_tree_models/                # Decision trees, ensembles, matrix compilation
├── ch08_unsupervised/               # Clustering, PCA, t-SNE
├── ch09_evaluation/                 # Metrics, cross-validation, complexity analysis
├── ch10_neural_networks/            # Neural nets from scratch, regularization
├── ch11_multi_model/                # Transfer learning, MTL, federated learning
├── ch12_dl_optimization/            # Mixed precision, gradient checkpointing
├── ch13_llms/                       # Memory profiling, LoRA, RAG
└── ch14_mlops/                      # Distillation, deployment, drift detection
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
cd data-science-lab
pip install -e ".[dev]"
```

### Running All Tests

```bash
pytest -v --tb=short
```

### Running Tests for a Specific Chapter

```bash
pytest ch01_programming_foundations/ -v
pytest ch06_linear_models/ -v
```

### Running a Specific Lab's Tests

```bash
pytest ch01_programming_foundations/lab1_data_cleaning/tests/ -v
```

## 📓 Jupyter Notebooks

Each lab includes a runnable Jupyter Notebook (`.ipynb`) with:
- Markdown explanations for each concept
- Step-by-step executable code cells
- Inline visualizations and analysis

```bash
jupyter notebook
# Navigate to any chapter → lab → .ipynb file
```

## 📋 Lab Overview

| Chapter | Labs | Topics |
|---------|------|--------|
| Ch 1 | 3 | Data cleaning, CSV analyzer, ETL pipeline |
| Ch 2 | 3 | Matrix ops, SVD compression, gradient descent |
| Ch 3 | 3 | Monte Carlo, Bayesian inference, A/B testing |
| Ch 4 | 4 | Feature engineering, preprocessing, time series, NLP |
| Ch 5 | 3 | Interactive dashboard, BI report, storytelling |
| Ch 6 | 3 | OLS from scratch, regularization, diagnostics |
| Ch 7 | 3 | Decision tree viz, ensemble benchmark, tree matrices |
| Ch 8 | 3 | Customer segmentation, PCA/t-SNE, dim reduction |
| Ch 9 | 3 | Evaluation framework, cross-validation, complexity |
| Ch 10 | 3 | NN from scratch, regularization ablation, dynamics |
| Ch 11 | 3 | Transfer learning, multitask, federated learning |
| Ch 12 | 3 | Mixed precision, checkpointing, accumulation |
| Ch 13 | 3 | LLM memory profiling, LoRA, RAG pipeline |
| Ch 14 | 3 | Knowledge distillation, deployment, drift detection |
| **Total** | **42** | |

## 📜 License

MIT
# data-science-lab
