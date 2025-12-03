# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 12:54:36 2025

@author: mpere
"""
# =============================================================================
# IMPORT
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import argparse
import pytensor
import pandas as pd



import nov_2025.upload_data as upload_data
import nov_2025.apoptosis_models as ap_models

# =============================================================================
# FUNCTIONS
# =============================================================================

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    df_fret = upload_data.load_data_and_phenotype(path_to_pickle = "C:/Users/mpere/Desktop/BHI_Apoptosis/BI_Apoptosis_data/dataset_train_BI_apoptosis_200.pkl",
                                      include_HPAF = False)