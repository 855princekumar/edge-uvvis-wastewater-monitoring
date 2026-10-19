# Credits and attribution

## Primary data source (must be cited)

All spectral and laboratory data analysed in this repository originate from the following
openly published dataset. It is the **primary source**; this repository is only a
**secondary analysis** of it.

> Lechevallier, P., Gruber, G., Bares, V., Neuenhofer, N., Waldner, L., Mahajan, A.,
> Mutzner, L., & Rieckermann, J. (2024). *Dataset on wastewater quality monitoring with
> adsorption and reflectance spectroscopy in the UV/Vis range* (Version 1.0). Eawag: Swiss
> Federal Institute of Aquatic Science and Technology. https://doi.org/10.25678/000D3C

- **Publisher / repository:** Eawag ERIC / opendata.eawag.ch
- **License (dataset):** Creative Commons Zero (CC0 1.0) - https://creativecommons.org/publicdomain/zero/1.0/ (DataCite record of 10.25678/000D3C). The data descriptor article (Lechevallier et al., 2025, *Scientific Data*) is CC BY 4.0.
- **Dataset contact:** Joerg Rieckermann, Eawag

### Attribution notice (good scholarly practice; CC0 does not require it)
This work includes / is derived from material by Lechevallier et al. (2024), Eawag,
released under CC0 1.0. Changes were made: the tabular UV-Vis absorbance spectra were
time-joined to laboratory grab-sample values and used to train and cross-validate
regression models. No endorsement by the original authors or Eawag is implied.

## Related peer-reviewed article
The dataset accompanies:

> Lechevallier, P., et al. (2025). Dataset on wastewater quality monitoring with adsorption
> and reflectance spectrometry in the UV-vis range. *Scientific Data*, 12(1), 1296.
> https://doi.org/10.1038/s41597-025-05459-x

## This repository
- Secondary analysis, code, figures, and results (c) 2025 Prince Kumar and Harshpreet Kaur.
- Code is released under the MIT License (see LICENSE). The MIT license applies to the code
  only and does **not** relicense the dataset.

## Acknowledgement of tools
Analysis pipeline built in Python (NumPy, pandas, scikit-learn, SciPy, PyTorch, Matplotlib);
interactive dashboards use Apache ECharts.
