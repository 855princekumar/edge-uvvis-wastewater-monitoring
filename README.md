# Edge-Enabled UV-Vis-Raman Wastewater Monitoring: code and results

[![Code license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC-BY 4.0](https://img.shields.io/badge/data-CC--BY%204.0-lightgrey.svg)](CREDITS.md)
[![Dataset DOI](https://img.shields.io/badge/dataset%20DOI-10.25678%2F000D3C-orange.svg)](https://doi.org/10.25678/000D3C)

Supplementary code and results for the manuscript **"Edge-Enabled UV-Vis-Raman Spectroscopy
for Real-Time Wastewater Monitoring."** This repository lets a reader reproduce the
secondary-data validation end-to-end. It contains **only** the analysis code and the results
we computed; the raw dataset is obtained from its original open source and is not
redistributed here.

## Abstract

Sewage treatment plants (STPs) are critical urban infrastructure, but their monitoring still
leans on laboratory workflows that sit awkwardly against real-time operational needs.
Laboratory methods such as UV-Visible spectrophotometry, HPLC, and Raman spectroscopy give
high chemical specificity and traceability, yet they depend on manual sampling, central
facilities, trained staff, and turnaround times of several days. Deployed Internet-of-Things
(IoT) sensor platforms solve the speed problem for bulk parameters (pH, dissolved oxygen,
turbidity) but cannot resolve molecular composition or reliably detect emerging contaminants.

This study brings together recent work on online spectroscopy, edge computing, and digital
compliance, and sets out a conceptual framework for edge-enabled UV-Vis-Raman monitoring at
STPs. Published laboratory studies and field deployments report that full-spectrum UV-Vis,
combined with chemometric or edge-based machine-learning models, can reach near-laboratory
accuracy (R2 above 0.9) for surrogate parameters in realistic monitoring. As a
proof-of-concept, we validated the machine-learning core of the framework on an open 25-week
wastewater dataset: UV-Vis spectra predicted ammonium at R2 ~ 0.85 (five-fold
cross-validation) on this limited external dataset, and a lightweight partial-least-squares
(PLS) model matched or exceeded an off-the-shelf 1D-CNN across all determinands tested, a
result directly relevant to low-cost edge deployment. The framework does not aim at
autonomous control. It keeps a human in the loop and treats the system as decision support,
predictive maintenance, and compliance-grade record-keeping aligned with ISO 15839 and
Central Pollution Control Board (CPCB) requirements. On the evidence reviewed, such systems
are feasible on embedded hardware (Raspberry Pi and Jetson class), scalable in cost, and
compatible with SCADA and telemetry already in use.

**Keywords:** sewage treatment plant (STP); real-time wastewater monitoring; UV-Vis-Raman;
edge computing; machine learning; chemometrics; CPCB compliance.

## What the code does

`run_experiment.py` reads the open Eawag wastewater dataset, time-aligns UV-Vis absorbance
spectra to laboratory grab-sample values, and benchmarks a 1D-CNN (proposed) against a PLS
regression (control) with 5-fold cross-validation, writing metrics, figures, and a report
per run.

## Data (obtain from primary source; not redistributed here)

> Lechevallier, P., Gruber, G., Bares, V., Neuenhofer, N., Waldner, L., Mahajan, A.,
> Mutzner, L., & Rieckermann, J. (2024). *Dataset on wastewater quality monitoring with
> adsorption and reflectance spectroscopy in the UV/Vis range* (v1.0). Eawag.
> https://doi.org/10.25678/000D3C - License: **CC-BY 4.0**.

Download only the **`2_data.zip`** resource from the DOI and place it beside
`run_experiment.py`. Details: [`data/README.md`](data/README.md). Full attribution:
[CREDITS.md](CREDITS.md).

## Reproduce

```bash
# raw spectra
python run_experiment.py --targets "nh4,po4,doc" --kfold 5 --tag raw
# smoothed + SNV
python run_experiment.py --targets "nh4,po4,doc" --kfold 5 --smooth --snv --tag smoothsnv
```

The script creates a local `.venv`, installs dependencies, and writes outputs to
`results/batch-N/`. Fixed seed (default 42). Without the real data file it runs on
clearly-labelled synthetic data stamped "NOT FOR PUBLICATION"; all committed results are
stamped `REAL (local 2_data.zip)`.

## Results (5-fold CV, mean +/- sd)

Smoothed + SNV configuration (`results/batch-2_smoothsnv/`):

| Target | PLS (control) | 1D-CNN (proposed) | RPD (PLS) |
|---|---|---|---|
| Ammonium (NH4) | 0.845 +/- 0.039 | 0.729 +/- 0.020 | ~3.0 |
| Phosphate (PO4) | 0.663 +/- 0.055 | 0.494 +/- 0.048 | ~1.9 |
| DOC | 0.243 +/- 0.426 | 0.237 +/- 0.087 | ~1.4 |

Raw-spectra results are in `results/batch-1_raw/`; a machine-readable comparison of all runs
is in [`results/batches_summary.csv`](results/batches_summary.csv). On this dataset a
lightweight PLS baseline matched or exceeded the 1D-CNN on every target, in both
configurations; DOC could not be reliably recovered from UV-Vis alone.

**Interactive dashboard:** <!-- ADD HOSTED DASHBOARD LINK HERE -->
_(hosted link to be added)_

The self-contained dashboard files are also included in the repository
([`results/batch-1_raw/dashboard.html`](results/batch-1_raw/dashboard.html),
[`results/batch-2_smoothsnv/dashboard.html`](results/batch-2_smoothsnv/dashboard.html)) and
can be opened locally in any browser.

## How to cite

Please cite **both** the primary dataset (DOI 10.25678/000D3C) and this repository; see
[CITATION.cff](CITATION.cff) and [CREDITS.md](CREDITS.md).

## License

Code: **MIT** ([LICENSE](LICENSE)). Data: **CC-BY 4.0**, (c) Lechevallier et al. (2024),
obtained from the source above and not redistributed here. The MIT license covers this
repository's code only.
