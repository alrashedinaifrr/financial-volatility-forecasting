# External GARCH estimation cross-check (Google Colab, 23 October 2026)

**Evidence source:** screenshot of user-executed `python src/verify_with_arch.py` on Google Colab, supplied in the conversation. Values were transcribed manually; this environment cannot install `arch` and has not independently executed the third-party estimator. The full screenshot is not included in the public distribution. The transcribed numbers are in `EXTERNAL_ARCH_RESULTS.csv`; the original screenshot is the primary evidence.

Both fits reported `ARCH optimizer converged: True`. S&P 500 fitted coefficients were closely aligned, with absolute common-objective gap about 1.28e-7. TASI coefficient differences were larger, with common-objective gap about 1.19e-4. The script computes the Gaussian quasi-likelihood for both parameter sets **under the custom implementation's initialization**, not each library's native log likelihood. Similar objective values do not prove identical forecasting performance or equality of estimated parameters.

**Status:** External estimator cross-check **executed and documented**; strict numerical equivalence and a complete cross-library forecast comparison are **not claimed**. This does not establish statistical superiority of any forecasting model.

**Reproduction on a connected machine:** `pip install arch` followed by `python src/verify_with_arch.py` from the project root (private data files required).
