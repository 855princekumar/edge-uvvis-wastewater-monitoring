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

These are the original v1.0.1 values (216 inputs, see the channel correction under v1.2.0 below); the
corrected 215-channel values used in the revised paper are in `results/batch-7_smoothsnv_215ch/` (ammonium PLS
0.860 +/- 0.026, 1D-CNN 0.743 +/- 0.018). Raw-spectra results are in `results/batch-1_raw/`; a machine-readable comparison of all runs
is in [`results/batches_summary.csv`](results/batches_summary.csv). On this dataset a
lightweight PLS baseline matched or exceeded the 1D-CNN on every target, in both
configurations; DOC could not be reliably recovered from UV-Vis alone.

**Interactive final results dashboard:**  [LINK](https://edge-uv-vis.855princekumar.workers.dev/) 

The self-contained dashboard files are also included in the repository
([`results/batch-1_raw/dashboard.html`](results/batch-1_raw/dashboard.html),
[`results/batch-2_smoothsnv/dashboard.html`](results/batch-2_smoothsnv/dashboard.html)) and
can be opened locally in any browser.

## v1.1.0 additions

### Study design: evidence first, hardware next

Every result in v1.1.0 carries its evidential status, so this repository can serve as the test
bed for the hardware and pilot stages that follow:

| Tier | What it is | What it tests | What it does not test |
|---|---|---|---|
| 1. Real data | Eawag raw-wastewater UV-Vis spectra + lab references, random and time-blocked CV | model accuracy on a real matrix; optimism of random CV | a built device; Indian STP influent/effluent |
| 2. Real hardware | `bench_edge.py` and `edge_node.py` on a physical Raspberry Pi 4 (4 GB) over a VPN | compute cost and end-to-end path on the target board | sensing, fouling, power |
| 3. Emulation | Pi-class Docker/QEMU profiles and an MQTT fleet replaying the real spectra | memory/CPU budgets and boards not available; loss; cross-architecture agreement | hardware timings (emulation overstated Pi 4 latency 9-30x) |
| 4. Simulation | `simulation/` physics-informed UV-Vis + Raman study | Raman hardware requirements under stated assumptions | validation of Raman fusion |

### Computational setup

| Machine | Hardware | Software |
|---|---|---|
| Workstation (x86-64) | 11th-gen Intel Core i7 laptop CPU (4 cores / 8 threads, 3.0 GHz base), 16 GB RAM, 512 GB NVMe SSD | Windows 11 Pro 64-bit; Tier 1 analysis in Python 3.14.6 (scikit-learn 1.9.0, NumPy 2.5.1, PyTorch 2.13.0 CPU); Docker Engine 29.5.3 in WSL2 (kernel 6.18, 8 vCPUs, 7.5 GiB), Debian 12 images for amd64/arm64/armv7, QEMU user-mode emulation, Mosquitto 2.1.2 |
| Edge node (physical) | Raspberry Pi 4 Model B rev 1.5, BCM2711 4x Cortex-A72 @ 1.8 GHz, 4 GB LPDDR4, 128 GB microSD | Raspberry Pi OS 64-bit (Debian 12, kernel 6.6.31), Python 3.11.2, NumPy 1.24.2, SciPy 1.10.1, scikit-learn 1.2.1, paho-mqtt 1.6.1; reached over an encrypted VPN (SSH for benchmarks, MQTT for the fleet test) |

### Time-blocked validation (leave-one-period-out)

Random folds can put temporally adjacent grab samples into both training and validation.
`--cv blocked` holds out five contiguous periods (about five weeks each, 102 samples), with the
inner PLS latent-variable selection also blocked.

```bash
# regression check: reproduces results/batch-2_smoothsnv exactly
python run_experiment.py --targets "nh4,po4,doc" --kfold 5 --smooth --snv --tag repro_smoothsnv
python run_experiment.py --spectral-only --targets "nh4,po4,doc" --kfold 5 --smooth --snv --tag smoothsnv_215ch
python run_experiment.py --spectral-only --targets "nh4,po4,doc" --kfold 5 --smooth --snv --cv blocked --tag smoothsnv_blocked_215ch
python run_experiment.py --spectral-only --targets "nh4,po4,doc" --kfold 5 --tag raw_215ch
python run_experiment.py --spectral-only --targets "nh4,po4,doc" --kfold 5 --cv blocked --tag raw_blocked_215ch
python block_spans.py      # date range of each blocked fold
```

CV R² (mean +/- sd), smoothed + SNV, 215 spectral channels (`results/batch-7_smoothsnv_215ch/`, `results/batch-8_smoothsnv_blocked_215ch/`):

| Target | PLS random | PLS blocked | 1D-CNN random | 1D-CNN blocked |
|---|---|---|---|---|
| Ammonium | 0.860 +/- 0.026 | 0.821 +/- 0.076 | 0.743 +/- 0.018 | 0.596 +/- 0.202 |
| Phosphate | 0.659 +/- 0.051 | 0.564 +/- 0.096 | 0.496 +/- 0.051 | 0.352 +/- 0.153 |
| DOC | 0.247 +/- 0.430 | -0.522 +/- 1.110 | 0.236 +/- 0.111 | -0.172 +/- 0.289 |

Holding out whole periods costs PLS little for ammonium but costs the CNN more, and DOC does
not generalise to unseen periods at all. Raw-spectra results: `results/batch-6_raw_215ch/`, `results/batch-9_raw_blocked_215ch/`.

**Channel correction (v1.2.0).** The column selection of v1.0.1/v1.1.0 also read the dataset's
`validity_test3_range_scan_absorbance2550_m` field as a 216th channel. `--spectral-only` keeps the 215
`scan_absorbance` channels; it is off by default so that v1.0.1 stays exactly reproducible, and all
v1.2.0 results and the paper use it (`results/batch-6..9_*_215ch`). PLS changed little; the untuned CNN on
raw spectra changed more. Batches 1-5 are the earlier 216-input runs, kept for the record.

### Edge inference benchmark and emulated Pi-class fleet (EMULATED, not Raspberry Pi hardware)

`bench_edge.py` times the actual models (PLS, 12 latent variables; the paper's 1D-CNN in
PyTorch if installed, and as a pure-NumPy forward pass that also runs on 32-bit ARM).
`fleet/` runs it in Docker containers for linux/arm64 and linux/arm/v7 under **QEMU
user-mode emulation** with Docker cgroup memory/CPU limits. The limits are real; the CPU
speed is the emulator's, not a Pi's, and is not proportional to it. The physical-Pi numbers
below were measured with
`python3 bench_edge.py --repeats 2000 --sustain 30 --label pi4-real --out bench_pi_real.json`.

```bash
python fleet/export_model.py --data-root data/raw/extracted   # PLS coefficients + real replay spectra
cd fleet
./run_bench_matrix_win.sh          # or run_bench_matrix.sh on Linux/macOS -> out/bench_*.json
COMPOSE_BUILD=--build ./run_fleet.sh 1 1800 my_run   # first run builds images; 7 nodes, MQTT,
                                                      # real spectra at 1 Hz for 30 min -> runs/my_run/
python plot_fleet.py runs/my_run figs
```

| Measurement | PLS (12 LV) | 1D-CNN (NumPy) | Source |
|---|---|---|---|
| **Physical Raspberry Pi 4, 4 GB**, median / p95 (ms) | 0.189 / 0.214 | 2.84 / 2.94 | `fleet/real_pi/bench_pi4_real_1thread_215ch.json` |
| Emulated arm64 4 GB profile, median (ms) | 1.63 | 86.5 | `fleet/out_bench_215/` |
| Native x86-64 container, median (ms) | 0.08 | 0.67 | `fleet/out_bench_215/` |

215-channel inputs; the other emulated profiles (`fleet/out_bench/`) and the fleet runs used the earlier 216-input model
(under 1% difference in computation).

All seven emulated profiles, down to 256 MB and half a core, completed without OOM. PyTorch
is not available on 32-bit ARM; the NumPy CNN covers it.

Fleet test (real spectra, MQTT QoS 1, 1 Hz per node = 120x the real 2-min rate):
- run A, 7 emulated nodes, 30 min: 0% loss on every node; median end-to-end latency 6.7 to
  448 ms; node predictions equal host predictions to within 6e-13 mg/L on x86-64, arm64 and
  armv7 (`fleet/runs/fleet_1hz_1800s/`);
- run B, the same plus the physical Pi 4 over a VPN, 10 min: 0% loss; Pi median end-to-end
  latency 192 ms (two VPN transits included), whole-board CPU about 4% of one core
  (`fleet/runs/fleet_with_real_pi_1hz_600s/`; `run_fleet_with_real_pi.sh`).

- run C, broker restarted at about 300 s of a 10-min run: **47% loss on every node**. The node
  client (`edge_node.py`) reconnects but does not renew its subscription, so nodes go silent
  after a broker restart. Recorded as a result; renewing subscriptions in `on_connect` and
  persistent sessions are required before the Stage 2 soak test
  (`fleet/runs/x_broker_restart_600s/`).

- run D, burst at 10 Hz per node (1,200x) for 5 min: five nodes returned every result; the two
  one-core arm64 profiles fell behind under emulation (66.6% and 49.7% not returned when the run
  closed) (`fleet/runs/x_burst_10hz_300s/`).
- memory floor 128 MB (`fleet/out_bench_x/`): armv7 ran (peak RSS at the limit); arm64 was
  OOM-killed (exit 137).

A 30-min run is not the 72 h soak test of Stage 2, so Stage 2 is only partially evaluated.

### In-silico UV-Vis + Raman design study (SIMULATION, not a validation)

`simulation/sim_fusion.py` and `simulation/analyse.py` generate synthetic nitrifying-effluent
UV-Vis and Raman spectra under stated assumptions and estimate when Raman would reduce
prediction error. Outputs: `simulation/sim_*.csv`, `simulation/fig_*.png`. These results are
conditional on the assumptions and are used only to set hardware requirements.

Robustness (v1.2.0): `simulation/robust_sensitivity.py` (Morris global sensitivity analysis with SALib, ten
assumptions, 220 parameter sets) and `simulation/robust_realmatrix.py` (nitrate and nitrite bands added to the
510 real Eawag spectra; export them first with `python simulation/export_real.py data/raw/extracted simulation/real510.npy`,
not committed). Results: a 10% nitrite gain needed SNR <= 50 in 65% and SNR <= 100 in 91% of parameter sets; on the real matrix,
UV-Vis alone gave nitrate R² 0.967 and nitrite R² 0.59, and Raman fusion reduced nitrite RMSE by
7% / 32% / 69% at SNR 50 / 100 / 300 (fluorescence-to-signal 3). Hence the Stage 3 specification SNR >= 100.

## How to cite

Software archive: v1.2.0 https://doi.org/10.5281/zenodo.23438582 (v1.1.0: https://doi.org/10.5281/zenodo.23434324; all versions:
https://doi.org/10.5281/zenodo.21904541).

Please cite **both** the primary dataset (DOI 10.25678/000D3C) and this repository; see
[CITATION.cff](CITATION.cff) and [CREDITS.md](CREDITS.md).

## License

Code: **MIT** ([LICENSE](LICENSE)). Data: **CC-BY 4.0**, (c) Lechevallier et al. (2024),
obtained from the source above and not redistributed here. The MIT license covers this
repository's code only.
