# V15 validation decision

**Closed:** The previously outstanding task *run an external GARCH estimator* is complete, based on the user's Google Colab screenshot. Both external optimizations reported convergence.

**Not claimed:** exact equivalence of parameter estimates, conditional variance paths, or forecasts; native likelihood equality; statistical superiority; data redistribution permission.

**Internal validation:** see `TEST_RUN.txt` for the fresh automated test output and `results_final/` for fresh model output.

## Current working-revision checks (28 October 2026)

- 15 automated tests pass, including three tests for paired QLIKE contrasts, Holm correction, and aligned-date validation.
- A synthetic end-to-end run generated the expected outputs, figures, and run metadata.
- `results_final/pairwise_qlike_comparisons.csv` was calculated from the repository's recorded row-level predictions and contains 60 model-pair contrasts.
- The actual market-price inputs are not included here, so the revised full runner has not been rerun on the original price files.
- Pairwise bootstrap intervals are pointwise and exploratory; Holm adjustment applies to the reported p-values only.
