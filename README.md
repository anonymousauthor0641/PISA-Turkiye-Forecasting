# Türkiye PISA Mathematics and Science Literacy Forecasting

This repository contains the source code, datasets, documentation, and figures associated with the study:

**Estimating PISA Mathematics and Science Literacy Scores Using Deep Learning Time Series Models: Integration of Socioeconomic Factors to the Models**

## Study overview

The study compares statistical and deep learning time-series forecasting approaches for Türkiye's officially reported national PISA mathematics and science literacy indicators. The analysis covers the 2003, 2006, 2009, 2012, 2015, 2018, and 2022 assessment cycles and produces exploratory forecasts for 2025, 2028, and 2031.

The forecasting framework includes:

- ARIMA
- Prophet
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Naive Persistence Baseline
- Expanding-window Rolling Backtesting
- Five-seed robustness analysis for LSTM and GRU
- Min–Max scaling fitted within each training fold
- Variance Inflation Factor (VIF) analysis
- Progressive integration of HEDRES, HOMEPOS, and WEALTH
- Historical and forecast figures for mathematics and science literacy

## Repository structure

```text
PISA_Turkiye_Forecasting_GitHub/
├── code/
│   ├── 01_data_preprocessing.py
│   ├── 02_vif_analysis.py
│   ├── 03_naive_baseline.py
│   ├── 04_rolling_backtesting.py
│   ├── 05_multiseed_analysis.py
│   ├── 06_future_forecasting.py
│   ├── 07_generate_figures.py
│   ├── 08_run_all.py
│   ├── models/
│   └── utils/
├── dataset/
├── documentation/
├── figures/
├── outputs/
├── requirements.txt
├── run_all.bat
├── run_all.sh
├── CITATION.cff
├── LICENSE
└── README.md
```

## Dataset

The analysis uses officially reported OECD country-level national mean PISA indicators for Türkiye rather than student-level microdata.

Dependent variables:

- Mathematics literacy score
- Science literacy score

Socioeconomic variables:

- `HEDRES`: Home Educational Resources
- `HOMEPOS`: Home Possessions
- `WEALTH`: Household Wealth Index

Missing socioeconomic observations were completed using nearest-year estimation and linear trend extension. Future socioeconomic inputs for 2025, 2028, and 2031 were generated through linear trend extrapolation and are used only for exploratory scenario-based forecasting.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Prophet and TensorFlow are optional but required to reproduce the Prophet, LSTM, and GRU outputs.

## Running the complete workflow

From the repository root:

```bash
python code/08_run_all.py
```

Windows users may run:

```text
run_all.bat
```

Linux/macOS users may run:

```bash
bash run_all.sh
```

Generated files are saved under `outputs/`.

## Main analytical stages

1. Data preprocessing and missing-value completion
2. Future socioeconomic input generation
3. VIF analysis
4. Naive Persistence baseline evaluation
5. Rolling backtesting
6. ARIMA, Prophet, LSTM, and GRU comparison
7. Five-seed robustness analysis
8. Progressive socioeconomic-variable integration
9. Exploratory forecasting for 2025, 2028, and 2031
10. Figure and table generation

## Interpretation and limitations

The repository supports an exploratory national-level forecasting study based on only seven PISA cycles. The results should not be interpreted as causal estimates, evidence of stable deep-learning superiority, exact forecasts of latent student proficiency, or evaluations of educational policy effects.

## Reproducibility

Random seeds, model configuration, forecast years, scenario definitions, and output paths are defined in `code/config.py`. Scalers are fitted within each rolling training fold to reduce data leakage.

## Citation

Citation metadata are available in `CITATION.cff`. Publication details can be updated after the article is accepted or published.

## License

The source code is released under the MIT License. Dataset reuse remains subject to the terms and attribution requirements of the original OECD PISA source.
