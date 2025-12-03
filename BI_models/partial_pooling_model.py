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
from pymc.ode import DifferentialEquation
import pytensor.tensor as pt



import nov_2025.upload_data as upload_data
import nov_2025.apoptosis_models as ap_models


# Define the ODE system
def sys_ing(t, y, R, l):
    dy = np.zeros(4)
    dy[0] = R - l*y[0]  # mA
    dy[1] = R/2 - l*y[1]  # mB
    dy[2] = R/5 - l*y[2]  # mC
    dy[3] = R/10 - l*y[3]  # mD
    return dy


# =============================================================================
# FUNCTIONS
# =============================================================================
def build_hierarchical_model_eaim(
    time_points, data, drug_dose, e_max, ic50, phenotype_id, clone_id, drug_id, kdeg_drug_dependent = True
):
    N_cells, T_len = data.shape
    N_pheno = len(np.unique(phenotype_id))
    N_clone = len(np.unique(clone_id))
    N_drug = len(np.unique(drug_id))
    
    
    def eaim_system_bayesian(y,t, theta ):
        """
        y: array of 10 variables
        % T, R, Z0, Z1,pC8, Z2, Z3, FLIP, C8, FRET

        params: array/list with at least 8 elements
        alphaR_3: fixed parameter (from MATLAB code, missing in snippet)
        """
      
        
        # Fixed parameters 
        K1_hat = rK1bK1 = 3.32807674e+01 # MATLAB params(3)
        K2_hat = rK2bK2 = 1.19478494e+02  # MATLAB params(4)
        K3_hat = rK3bK3 = 2.60451297e+01  # MATLAB params(5)
        K4_hat = rK2K1 = 4.96917812e+00  # MATLAB params(6)
        K5_hat = rK3K1 = 4.36614979e-03  # MATLAB params(7)
        alpha2 = rK_fret = 3e-4  # MATLAB params(8)
        alphaR_3 = 5.04995424e+01
        rK_fret = 3.0046e-04
        alphaC8 = 8.490566037735849
        
        T,R, Z0, Z1, pC8, Z2, Z3, FLIP, C8, FRET = y[0], y[1],y[2],y[3], y[4], y[5], y[6], y[7], y[8], y[9]

        
        alpha0_tilde, alpha1_tilde, K_deg = theta[0], theta[1], theta[2]

        # Equations
        # attention 
        # Z3^c = Z2^e
        # Z2^c = Z1^e
        # Z1^c = Z3^e
        # Trail T
        T_new = -((T * R**3) / (R**3 + alphaR_3)) + rK1bK1 * Z0
        
        # Receptor R
        R_new = -3*((T * R**3) / (R**3 + alphaR_3)) + 3*rK1bK1 * Z0
        
        # Z0 = T:R^3 
        Z0_new = ((T * R**3) / (R**3 + alphaR_3)) - rK1bK1*Z0 - rK3K1*Z0*FLIP**3 +\
            rK3bK3*rK3K1*Z1 - rK2K1*Z0*pC8**2 + rK2bK2*rK2K1*Z2 + alpha0_tilde*Z2
            
        # Z1 = 
        Z1_new = rK3K1*Z0*FLIP**3 - rK3bK3*rK3K1*Z1
        # pC8
        pC8_new = -2*rK2K1*Z0*pC8**2 + 2*rK2bK2*rK2K1*Z2
        # Z2 = FADD = T:R^3:pC8^2
        Z2_new = rK2K1*Z0*pC8**2 - rK2bK2*rK2K1*Z2 - rK2K1*Z2 *FLIP + rK2bK2*rK2K1*Z3\
            - alpha0_tilde*Z2 
        # Z3 = ?
        Z3_new = rK2K1*Z2 *FLIP - rK2bK2*rK2K1*Z3 - alpha1_tilde*Z3
        # FLIP
        FLIP_new = -3*rK3K1*Z0*FLIP**3 + 3*rK3bK3*rK3K1*Z1 - rK2K1*Z2*FLIP + rK2bK2*rK2K1*Z3
        # C8
        C8_new = alpha0_tilde*Z2 + alpha1_tilde*Z3 - K_deg*(C8/(alphaC8 + C8)) - rK_fret*C8
        # FRET
        FRET_new = rK_fret*C8  # last line from MATLAB code
        
        return [T_new, R_new, Z0_new,Z1_new, pC8_new, Z2_new, Z3_new,FLIP_new, C8_new, FRET_new]


    eaim_solver = DifferentialEquation(
        func = eaim_system_bayesian,
        times=time_points,
        n_states=10,
        n_theta=3,
        t0=0
    )

    with pm.Model() as model:

        # =======================================================
        # 1. Population-level hyperpriors
        # =======================================================
        # phenotype-level parameters hyperpriors
        μ_Kdeg  = pm.Normal("μ_Kdeg",  0.2, 0.1)
        μ_FLIP0 = pm.Normal("μ_FLIP0", 0.0, 1.0)
        μ_pC80  = pm.Normal("μ_pC80",  0.0, 1.0)

        σ_Kdeg  = pm.HalfNormal("σ_Kdeg",  0.05)
        σ_FLIP0 = pm.HalfNormal("σ_FLIP0", 0.5)
        σ_pC80  = pm.HalfNormal("σ_pC80", 0.5)

        # clone-level parameter hyperpriors
        μ_alpha0 = pm.Normal("μ_alpha0", 0.05, 0.02)
        μ_alpha1 = pm.Normal("μ_alpha1", 0.1,  0.02)
        μ_FRET0  = pm.Normal("μ_FRET0",  0.0,  0.5)

        σ_alpha0 = pm.HalfNormal("σ_alpha0", 0.02)
        σ_alpha1 = pm.HalfNormal("σ_alpha1", 0.02)
        σ_FRET0  = pm.HalfNormal("σ_FRET0",  0.2)
        
        
        # drug dose-level parameter hyperpriors
        μ_Kdeg_dd  = pm.Normal("μ_Kdeg_dd",  0.2, 0.1)
        μ_FLIP0_dd = pm.Normal("μ_FLIP0_dd", 0.0, 1.0)
        μ_pC80_dd  = pm.Normal("μ_pC80_dd",  0.0, 1.0)

        σ_Kdeg_dd  = pm.HalfNormal("σ_Kdeg_dd",  0.05)
        σ_FLIP0_dd = pm.HalfNormal("σ_FLIP0_dd", 0.5)
        σ_pC80_dd  = pm.HalfNormal("σ_pC80_dd", 0.5)
       
        # =======================================================
        # 2. Phenotype-level biological parameters
        # =======================================================
        Kdeg_ph  = pm.Normal("Kdeg_ph",  μ_Kdeg,  σ_Kdeg,  shape=N_pheno)
        FLIP0_ph = pm.Normal("FLIP0_ph", μ_FLIP0, σ_FLIP0, shape=N_pheno)
        pC80_ph  = pm.Normal("pC80_ph",  μ_pC80,  σ_pC80,  shape=N_pheno)

        # =======================================================
        # 3. Clone-level parameters
        # =======================================================
        alpha0_cl = pm.Normal("alpha0_cl", μ_alpha0, σ_alpha0, shape=N_clone)
        alpha1_cl = pm.Normal("alpha1_cl", μ_alpha1, σ_alpha1, shape=N_clone)
        FRET0_cl  = pm.Normal("FRET0_cl",  μ_FRET0,  σ_FRET0,  shape=N_clone)
        
        # =======================================================
        # 4. Drug-level 
        # =======================================================
        Kdeg_dd  = pm.Normal("Kdeg_dd",  μ_Kdeg_dd,  σ_Kdeg_dd,  shape=N_drug)
        FLIP0_dd = pm.Normal("FLIP0_dd", μ_FLIP0_dd, σ_FLIP0_dd, shape=N_drug)
        pC80_dd  = pm.Normal("pC80_dd",  μ_pC80_dd,  σ_pC80_dd,  shape=N_drug)

        # =======================================================
        # 5. Weak cell-level offsets
        # =======================================================
        σ_offset = pm.HalfNormal("σ_offset", 0.1)

        δKdeg  = pm.Normal("δKdeg",  0, σ_offset, shape=N_cells)
        δFLIP0 = pm.Normal("δFLIP0", 0, σ_offset, shape=N_cells)
        δpC80  = pm.Normal("δpC80",  0, σ_offset, shape=N_cells)

        # Combine drug_dose + phenotype + offsets
        dose_i = pt.take(drug_dose, drug_id)
        emax_i = pt.take(e_max, drug_id)
        ic50_i = pt.take(ic50, drug_id)
        
        
        emax_fct = 1 - (dose_i * emax_i) / (dose_i + ic50_i)
        if kdeg_drug_dependent:
            Kdeg_cell  = Kdeg_dd[drug_id] * emax_fct  + Kdeg_ph[phenotype_id]  + δKdeg
        else:
            Kdeg_cell  = Kdeg_ph[phenotype_id]  + δKdeg
        FLIP0_cell = FLIP0_dd[drug_id]* emax_fct + FLIP0_ph[phenotype_id] + δFLIP0
        pC80_cell  = pC80_dd[drug_id] * emax_fct + pC80_ph[phenotype_id]  + δpC80

        # Clone mappings
        alpha0_cell = alpha0_cl[clone_id]
        alpha1_cell = alpha1_cl[clone_id]
        FRET0_cell  = FRET0_cl[clone_id]

       

        # =======================================================
        # 6. Solve EAIM per cell
        # =======================================================
        yhat_list = []

        for i in range(N_cells):

            params_i = pt.concatenate([
                alpha0_cell[i:i+1],   # clone-level α0
                alpha1_cell[i:i+1],   # clone-level α1
                Kdeg_cell[i:i+1],     # phenotype + offset
            ])

            # initial conditions include FRET0 (clone-level)
            y0_i = pt.stack([
                drug_dose[i]*31.0, 52000,
                0.0, 0.0, 
                pC80_cell[i],  
                0.0, 0.0,
                FLIP0_cell[i],        # phenotype + offset
                30,
                FRET0_cell[i],        # clone-level pooled baseline
            ])

            sol = eaim_solver(y0=y0_i, theta=params_i)
            yhat_list.append(sol[:, -1])

        y_hat = pt.stack(yhat_list)

        # =======================================================
        # 7. Likelihood (NaNs are ignored automatically)
        # =======================================================
        σ = pm.HalfNormal("σ", 0.1)
        pm.Normal("obs", mu=y_hat, sigma=σ, observed=data)

    return model




# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    #load data
    df_fret = upload_data.load_data_and_phenotype(path_to_pickle = "/home/mpere/Desktop/Marielle_OneDrive/03_Data/BI_Apoptosis/dataset_train_BI_apoptosis_200.pkl",
                                                  include_HPAF = False)
    
    
    df_fret = df_fret.iloc[:100,:]
   
    # define time
    time_cols = [c for c in df_fret.columns if isinstance(c, (int, float))]
    time_points = np.array(time_cols)
    
    
    # define drug dose
    drug_dose = np.array(df_fret['Dose'], dtype = float)
    ic50 = np.array(df_fret['IC50'])
    e_max = np.array(df_fret['Emax'])
  
    
    
    # define categories
    clone_labels = df_fret["Clone"].astype("category")
    clone_id = clone_labels.cat.codes.values
    
    cell_line_labels = df_fret["Cell Line"].astype("category")
    cell_line_id = clone_labels.cat.codes.values
    
    df_fret['clone_drug_phenotype'] = df_fret['Clone']+'_'+df_fret['Dose']+'_'+df_fret['phenotype']
    df_fret['cellline_drug_phenotype'] = df_fret['Cell Line']+'_'+df_fret['Dose']+'_'+df_fret['phenotype']
    
    df_fret['clone_drug'] = df_fret['Clone']+'_'+df_fret['Dose']
    df_fret['cellline_drug'] = df_fret['Cell Line']+'_'+df_fret['Dose']
    
    
    phenotype_labels = df_fret["clone_drug_phenotype"].astype("category")
    phenotype_id = phenotype_labels.cat.codes.values
    
    drug_labels = df_fret["clone_drug"].astype("category")
    drug_id = phenotype_labels.cat.codes.values
    
      
    
    
    # extract into N_cells × T matrix
    data = df_fret[time_cols].to_numpy()
    
   
    
    # Build Model
    model = build_hierarchical_model_eaim(
        time_points=time_points,
        data=data,
        clone_id=clone_id,
        phenotype_id=phenotype_id,
        drug_dose = drug_dose,
        e_max = e_max, 
        ic50=ic50,  
        drug_id = drug_id, 
        kdeg_drug_dependent = True

    )
    
    # Run inference
    with model:
        trace = pm.sample(tune=500, draw=1000, cores=4, chains=4)
        
    summary = az.summary(trace, hdi_prob=0.95)
    print(summary)
    
  
    
   