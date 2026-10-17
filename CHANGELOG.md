# Changelog

## v1.2.0 (2026-10-17)

### Fixed
- Column selection: v1.0.1/v1.1.0 also read `validity_test3_range_scan_absorbance2550_m` (a validity-test
  field, not a spectral channel) as a 216th channel. New opt-in flag `run_experiment.py --spectral-only` keeps the
  215 `scan_absorbance` channels (default unchanged so v1.0.1 stays exactly reproducible; verified).
  `fleet/export_model.py` and `bench_edge.py` now use 215 channels.

### Added
- Corrected results with 215 channels: `results/batch-6_raw_215ch`, `batch-7_smoothsnv_215ch`,
  `batch-8_smoothsnv_blocked_215ch`, `batch-9_raw_blocked_215ch` (the paper uses these).
- 215-channel benchmarks: physical Raspberry Pi 4 (`fleet/real_pi/bench_pi4_real_1thread_215ch.json`), native and
  emulated arm64 4 GB (`fleet/out_bench_215/`); re-exported model `fleet/models/pls_nh4.json` (12 LV, 215 inputs).
- Robustness of the in-silico design study: `simulation/robust_sensitivity.py` (Morris/SALib, 220 parameter sets),
  `simulation/robust_realmatrix.py` (bands added to the real Eawag spectra), `simulation/export_real.py`,
  `simulation/robust_plot.py`, outputs and `fig_sim_robustness.png/.svg`. Outcome: Raman specification raised from
  SNR 50 to SNR 100 (fluorescence-to-signal 3). Simulation and semi-synthetic, not hardware.

## v1.1.0 (2026-10-17)

Revision release accompanying manuscript CSCEE-D-26-00739 (R1). The default behaviour of
`run_experiment.py` is unchanged: re-running the v1.0.1 command reproduces
`results/batch-2_smoothsnv` exactly (`results/batch-3_repro_smoothsnv`). The archived v1.0.1
results (`results/batch-1_raw`, `results/batch-2_smoothsnv`) are untouched.

### Added
- `run_experiment.py --cv blocked`: time-blocked five-fold cross-validation (contiguous time
  periods, leave-one-period-out; the inner PLS latent-variable selection is blocked too). Logs
  the date span, median gap between grab samples and per-fold R² and latent variables.
- Results: `results/batch-3_repro_smoothsnv` (regression check), `results/batch-4_smoothsnv_blocked`,
  `results/batch-5_raw_blocked`; `block_spans.py` writes the date range of each blocked fold
  (`results/batch-4_smoothsnv_blocked/block_date_spans.csv`).
- `bench_edge.py`: inference benchmark of the actual models (PLS with 12 latent variables; the
  1D-CNN of the paper in PyTorch where available and as a pure-NumPy forward pass that also
  runs on 32-bit ARM): median/p95 latency, peak memory, serialised size, throughput and CPU.
- `fleet/`: Docker emulator of a Raspberry-Pi-class edge fleet (Debian bookworm images for
  amd64, arm64 and armv7 under QEMU, Docker cgroup memory/CPU limits, Mosquitto MQTT), with
  model export (`export_model.py`), real-spectrum replay, collector, analysis, plots, the
  benchmark matrix (`run_bench_matrix.sh`, Windows-safe `run_bench_matrix_win.sh`) and an
  end-to-end runner (`run_fleet.sh`). Results in `fleet/out_bench/` and `fleet/runs/`.
  **ARM timings in `fleet/out_bench/` and of the emulated nodes are emulated, not Raspberry Pi
  measurements** (physical-Pi results are in `fleet/real_pi/`).
- Physical-hardware results: `bench_edge.py` on a Raspberry Pi 4 Model B (4 GB) reached over a
  VPN (`fleet/real_pi/`), and `run_fleet_with_real_pi.sh`, which adds a physical node to the
  emulated fleet through a published broker port (`EXTRA_NODES`, `PUBLISH_BROKER` in
  `gen_compose.py`).
- `fleet/analyse_extra.py` (container CPU, loss and recovery after a broker restart) and
  `fleet/plot_fleet.py` (Figs. S7-S8 of the paper).
- `simulation/`: physics-informed in-silico UV-Vis + Raman design study (`sim_fusion.py`,
  `analyse.py`) with its outputs. **Simulation only, not a validation.**

### Fixed
- `fleet/gen_compose.py`: environment entries are now quoted; previously the comma-separated
  `NODES` list was split by YAML, so the replayer published to the first node only.
- `fleet/gen_compose.py`: replayer warm-up raised to 90 s so slow emulated nodes subscribe
  before replay starts.
- `fleet/export_model.py`: replay spectra are filtered to `valid_data == 1`, as in training.
- `bench_edge.py`: also reports throughput/CPU and serialised size for the NumPy CNN.

### Changed
- `.gitignore` also excludes `fleet/out/` and `fleet/models/*.npy` (the exported replay spectra
  are derived from the CC-BY dataset and are regenerated locally).
- `CITATION.cff`: version 1.1.0.
