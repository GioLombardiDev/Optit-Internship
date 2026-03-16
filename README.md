# Improving Heat-Load Forecasting with Deep Learning (MSc Thesis)

This repository contains the code developed during an industry internship at [Optit Srl](https://optit.net/) on short-term heat-load forecasting. The work carried out during the internship was subsequently documented and further developed in my MSc thesis in Mathematics at the University of Bologna.

Thesis (PDF): [docs/thesis.pdf](docs/thesis.pdf)

The repository includes a simple `pyproject.toml` configuration so that the `heat_forecast` package can be installed locally. From the project root, run:

~~~bash
pip install .
~~~

## Thesis abstract

This thesis investigates short-term heat-load forecasting in the context of an internship at Optit, a company that develops decision-support systems across diverse sectors. The aim is to evaluate forecasting models as potential alternatives to those currently used in the proprietary [OptiEPTM](https://optit.net/soluzioni/) platform, namely XGBoost and Support Vector Regression, and to assess their potential to improve forecasting accuracy and the consistency of performance across horizons and series of varying difficulty.

The analysis is conducted on five synthetic hourly heat-demand series, representative of different application contexts, and considers two forecasting horizons of operational interest: day-ahead and week-ahead. A per-site modelling strategy is adopted, with each model optimised separately for each series and horizon. The models examined include MSTL-ETS, SARIMAX, LSTM, the Temporal Fusion Transformer, and the foundation model Chronos-2 in a zero-shot setting. The evaluation is based on rolling cross-validation, seasonal analysis, residual diagnostics, and bootstrap significance tests that account for temporal dependence.

The results show that there is no universally dominant model. On the more regular and structured series, LSTM emerges as the most reliable solution across both forecasting horizons. On the noisier and more challenging series, Chronos-2 proves particularly competitive, especially in the day-ahead setting. XGBoost nevertheless remains a strong benchmark thanks to its high computational efficiency, while SARIMAX offers greater interpretability despite weaker performance during transitional periods.

Overall, the study shows that improvements in forecasting performance depend on a trade-off between accuracy, computational cost, scalability, and interpretability, and provides a methodological and empirical basis for the further development of the forecasting module within OptiEPTM.

## Data

The internship dataset is confidential and is not included in this repository.

As a result, the repository does not provide full end-to-end reproducibility out of the box: running the notebooks requires user-provided data, and some analyses also assume locally generated intermediate outputs.

## Repository structure

Tracked in this repository:
- `docs/`: thesis PDF (`thesis.pdf`)
- `notebooks/`: exploratory analysis and modelling notebooks
- `src/heat_forecast/`: Python modules for data handling, modelling, and evaluation
- `pyproject.toml`: project metadata and package configuration

Expected local folders (represented by `.keep` placeholders and ignored by design):
- `data/`: user-provided datasets, including `data/timeseries/`
- `models/`: saved model parameters
- `results/`: training and testing outputs, including runtime metrics
- `logs/`: training and experiment logs

## Running the notebooks

To run the notebooks end-to-end, you will need to provide your own dataset. Some notebooks also assume that intermediate outputs have already been generated and saved locally. The repository includes the expected folder structure via `.keep` placeholder files, while data and generated artefacts are ignored by design.

You can run the notebooks in two ways.

### 1. Google Colab

Place the project folder `heat-forecast` in your Google Drive.  
The setup cell in each notebook mounts Drive and automatically adds `MyDrive/heat-forecast/src` to `sys.path`, allowing direct import of `heat_forecast`.

### 2. Local machine

Two options are available:

- **Install the package:** from the project root, run

  ~~~bash
  pip install .
  ~~~

  This installs the package normally, so notebooks can import `heat_forecast` without manually modifying `sys.path`.

- **Without installing:** if you run a notebook from `.../heat-forecast/notebooks/`, the setup cell detects `../src` and adds it to `sys.path` automatically.

## Licence

This repository is made publicly available for viewing and personal reference purposes only, as part of a personal academic portfolio.

You may view and download the contents for personal reference only. No licence is granted to modify, redistribute, republish, sublicense, or otherwise reuse the code or other materials in this repository without prior written permission from the author. All rights reserved.





