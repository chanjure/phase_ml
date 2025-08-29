% Arxiv badge here

[![Run tests](https://github.com/chanjure/phase_ml/actions/workflows/pytest.yaml/badge.svg?event=push)](https://github.com/chanjure/phase_ml/actions/workflows/pytest.yaml)
[![codecov](https://codecov.io/gh/chanjure/phase_ml/graph/badge.svg?token=D4Q9HTV7SW)](https://codecov.io/gh/chanjure/phase_ml)

# Phase diagram and eigenvalue dynamics of stochastic gradient descent in multilayer neural networks

Data and code release for [arxiv number]

## Repository Structure

```
.
├── data                            <- Trained models used in the paper
├── data_assets                     <- Directory for output of the workflow
│   ├── figures
│   ├── logs
│   └── models
├── env                             <- Environment files
├── libs                            <- Submodules including figure styles
├── LICENSE
├── README.md
└── src
    ├── bin                         <- Python source code for generating trained models
    │   └── generate.py             
    ├── notebooks                   <- Jupyter notebooks for reproducing the figures in the paper
    │   └── empirical_phase.ipynb
    └── scripts                     <- Scripts for running the workflow
        └── submit_train.sh
```

## Requirements

- Numpy
- Scipy
- Matplotlib
- Jupyter

## Setup

1. Install the dependencies above.

2. Clone this repository including submodules (or download its Zenodo release and ```unzip``` it) and ```cd``` into it:

```
git clone --recurse-submodules https://github.com/chanjure/phase_ml.git
cd phase_ml
```

3. Set up the environment

For conda users, you can create a new environment with the following command:

```conda env create -f env/environment.yml -n qftml```

Then, activate the environment:

```conda activate qftml```

For pip users, you can install the required packages with the following command:

```python -m pip install -r env/requirements.txt```

## Running the workflow

### Reproducing the plots

The plots in the article can be reproduced by the jupyter notebook in ```src/notebooks```.

### Regenerating the trained models

The models saved in ```data/fixed_eps_Ttf``` are generated from the training script in ```src/scripts/submit_train.sh```.

Run:

```bash ./src/scripts/submit_train.sh```

to regenerate trained models.

Hyperparameters such as learning rate, batch size, number of nodes, etc. can be modified in the script.

## Output

Output plots from the jupyter notebook are placed in the ```data_assets/figures``` directory.

Generated trained models are placed into the ```data_assets/models``` directory.

## Reusability

This workflow is relatively tailored to the data which it was originally written to analyse.
Additional data may be added to the analysis by adding relevant files to the ```data``` directory.
However, extending the analysis in this way has not been as fully tested as the rest of the workflow, and is not guaranteed to be trivial for someone not already familiar with the code.
