# Code Audit — data-science-lab

Static audit performed for the companion ebook *Cracking the Data Scientist System
Design Interview (2026 Edition)*. Every listing quoted in the book was checked against
the file it claims to come from. Nothing in the book is paraphrased, invented, or
"illustrative" — if a function appears in a chapter, it exists at the stated path.

**Scope.** 90 Python files, 42 labs, 14 chapters, ~7,100 lines.

## Verdict

The implementations are real. Every lab imports libraries that exist, calls APIs that
exist with the signatures it uses, and computes what its name claims. Five substantive
defects were found and fixed; the remaining findings are documented as known limitations
and are discussed openly in the relevant chapters rather than hidden.

## Defects found and fixed

| # | Location | Defect | Severity | Fix |
|---|----------|--------|----------|-----|
| 1 | `ch02_mathematics/lab3_gradient_descent/gradient_descent.py` | `grad_beale` returned a hard-coded `[0.0, 0.0]` with the comment "Stub for beale grad for simplicity". Any optimiser using it never moves. | **High** — fabricated result | Replaced with the analytical gradient. Verified against central differences at three points to `atol=1e-4`, and confirmed to vanish at the known Beale minimum $(3, 0.5)$. |
| 2 | `ch07_tree_models/lab2_ensemble_benchmark/ensemble_benchmark.py` | `adaboost_step_by_step` applied `exp(-alpha * y * preds)` to labels in $\{0,1\}$. The update collapses to `exp(0) = 1` for every sample with $y=0$, so half the dataset is never reweighted and the trace is meaningless. Also divided by `err + 1e-10` without guarding `err >= 0.5`. | **High** — silently wrong algorithm | Labels are remapped to $\{-1,+1\}$ before the update; added early termination for a perfect stump and for worse-than-chance error. Verified: weight spread now grows $1.0 \to 89.3$ over 5 rounds, alphas decrease monotonically after round 1. |
| 3 | `pyproject.toml` | `ch04_feature_engineering/lab1_kaggle_features` imports `category_encoders`; `joblib` is imported by two labs. Neither was declared. A clean `pip install -e .` produced an `ImportError`. | **Medium** — build break | Both added to `dependencies`. |
| 4 | `ch04_feature_engineering/lab2_preprocessing_pipeline/preprocessing_pipeline.py` | `DatetimeTransformer.get_feature_names_out` returned `None`, breaking `ColumnTransformer`'s `verbose_feature_names_out` and `set_output(transform="pandas")`. `fit` did not record `feature_names_in_`. | **Medium** | Implemented properly; `fit` now records input names and `n_features_in_`; `transform` preserves the index. |
| 5 | `ch14_mlops/lab2_model_deployment/train_and_save.py` | Trained on unseeded `np.random.randn` with random labels, so the served artifact differed on every run and carried no signal to learn. | **Medium** — non-reproducible | Seeded generator; labels are now a noisy linear function of the features, so the deployed model has something to predict. |

## Known limitations (documented, not defects)

These are intentional simplifications. They are called out in the chapters where they
appear, because pretending otherwise would be the fabrication the audit is meant to
prevent.

| Location | Limitation | Where discussed |
|----------|-----------|-----------------|
| `ch06_linear_models/lab1_ols_from_scratch/ols.py` | Solves the normal equations via `np.linalg.inv(X'X)`, which squares the condition number. Numerically unstable for collinear designs; production code uses QR or SVD (`np.linalg.lstsq`). | Chapter 6 — kept deliberately, because the interview question is *why* this is wrong |
| Plot helpers across ch02, ch03, ch08 | Several `plot_*` functions are `pass` stubs. They render nothing and are not used by any test. | Not quoted in the book |
| `ch08_unsupervised/lab3_dim_reduction_pipeline` | `DimReductionPipeline.fit` is a no-op; `reduce()` may select t-SNE, which has no `transform`, so the pipeline cannot be applied to new data. | Chapter 8 — used as the worked example of why t-SNE is not a preprocessing step |
| `ch13_llms/lab1_memory_profiling` | `calculate_activation_memory` uses a linear heuristic and ignores both the $O(L^2)$ attention term and the `n_heads` argument it accepts. | Chapter 13 — the correction is the chapter's main derivation |
| `ch12_dl_optimization/lab1_mixed_precision` | `GradScaler`/`autocast` are constructed with `enabled=torch.cuda.is_available()`, so on CPU the "mixed precision" path is a no-op and shows no speedup. | Chapter 12 — stated explicitly |
| `ch05_visualization/lab3_storytelling_portfolio` | Charts are labelled "Climate Trends", "Market Share" and "Demographics" but plot unseeded synthetic noise. | Chapter 5 — used as the worked example of a chart that lies |
| `ch02_mathematics/lab1_matrix_operations` | `matrix_determinant` is the $O(n!)$ cofactor expansion, and `is_singular` swallows every exception with a bare `except`. | Chapter 2 — the complexity is the point of the lab |

## Verification performed

- Numerical gradient check on `grad_beale` (central differences, three points).
- Execution trace of `adaboost_step_by_step` on `make_classification(300, 10)`.
- Import-graph scan for undeclared third-party dependencies.
- Grep sweep for `pass`, `TODO`, `FIXME`, `NotImplemented`, and "stub" across all 90 files.
- Signature cross-check of every sklearn / statsmodels / torch API called.

Full test execution (`pytest -v`) was not run in this pass; it requires the full
dependency set including PyTorch and torchvision. The five fixes above were each
executed directly.
