# bayesian_apoptosis_code
Code linked to "Exploring the kinetics determinants of TRAIL sensitivity states in multiple cancer cell lines through quantitative systems modeling and bayesian optimization identifies key-features of drug-tolerants persisters in single-cell experiments".

## Quick introduction
This python code run on Python 3.9 and is design to apply hierarchical bayesian calibration and classic bayesian fit of mathematical models using isogenic single-cell FRET trajectories from live-cell microscopy of multiple human cancer cell lines treated with different doses of TRAIL, a death ligand, to identify discriminative patterns in sensitive-state cell population compared to drug-tolerant ones.

This code is organised as follow with 3 folders:
* [data_management](data_management/): 
* [simulation_and_fit](simulation_and_fit/)
* [result](result/)
 
with a [hierarchical_modeling_main.py](hierarchical_modeling_main.py), a [data_selection.xlsx](data_selection.xlsx) used to locate input folders, a [config_parameters.ini](config_parameters.ini) with all necessary parameters for simulation and a [requirements.txt](requirements.txt) for installation.

## Installation
Open a terminal in bayesian_apoptosis_code folder. Run the following code:
```conda create -c conda-forge -n pymc_env python=3.10 "pymc>=5"
    conda activate pymc_env
    conda install --name pymc_env --file requirements.txt
    conda install -n pymc_env conda-forge::concurrent-log-handler
```


