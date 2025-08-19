% Badges here

# Phase diagram and eigenvalue dynamics of stochastic gradient descent in multilayer neural networks

Data and code release for [To be filled]

## Repository Structure

```
├── data
│   └── fixed_eps_Ttf           <- Trained models used in the paper
│       └── README.txt
├── libs                        <- Matplotlib style files
│   └── global_chanju
├── LICENSE
├── README.md
└── src
    ├── bin                     <- Python source code for generating trained models
    │   └── generate.py
    ├── notebooks               <- Jupyter notebooks for analysis and visualization
    └── scripts
        └── submit_train.sh     <- bash scripts for running source codes
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

The plots in the article can be reproduced by Jupyter notebooks in ```src/notebooks```.

### Regenerating the trained models

The models saved in ```data/fixed_eps_Ttf``` are generated from the training script in ```src/scripts/submit_train.sh```.

Run from ```src/scripts``` directory:

```bash bash submit_train.sh```

To regenerate trained models.

## Output

Output plots are placed in the ```assets/plots``` directory.

Output data assets are placed into the ```data_assets``` directory.

## Reusability

This workflow is relatively tailored to the data which it was originally written to analyse.
Additional data may be added to the analysis by adding relevant files to the ```data``` directory.
However, extending the analysis in this way has not been as fully tested as the rest of the workflow, and is not guaranteed to be trivial for someone not already familiar with the code.
