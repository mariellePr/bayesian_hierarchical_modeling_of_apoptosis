# bayesian_apoptosis_code
Code linked to "Bayesian modeling of apoptosis decomposes TRAIL-response dynamic signature of drug-tolerant cells across multi-level cell hierarchy in Gastro-Intestinal cancers".

## Installation
Open a terminal in bayesian_apoptosis_code folder. Run the following code:
```conda create -c conda-forge -n pymc_env python=3.10 "pymc>=5"
    conda activate pymc_env
    conda install --name pymc_env --file requirements.txt
    conda install -n pymc_env conda-forge::concurrent-log-handler
```


