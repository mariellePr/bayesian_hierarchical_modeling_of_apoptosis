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
import configparser



import upload_data as upload_data
import apoptosis_models as ap_models




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

def trim_outliers(series, q=0.10):
    """Remove the lowest and highest q fraction of data."""
    low = series.quantile(q)
    high = series.quantile(1 - q)
    return series[(series >= low) & (series <= high)]


def build_hierarchical_model_FRETexp(
    time_points, data, drug_dose, e_max, ic50, phenotype_id, clone_id, drug_id,
    beta0_drug_dependent = True
):
    N_cells, T_len = data.shape
    N_pheno = len(np.unique(phenotype_id))
    N_clone = len(np.unique(clone_id))
    N_drug = len(np.unique(drug_id))
    
    
    mask_nan = ~np.isnan(data)
    
    time_matrix = np.ones((N_cells,len(time_points)))* time_points
    time_points_without_nan = time_matrix[mask_nan]
  
   
 
    
    with pm.Model() as model:
        pm.Data("time_vector_without_nan", time_points_without_nan,\
             dims =["cellsxtime_id"])
        # pm.Data("FRET", data,dims =["cells", "time_id"])
        pm.Data("FRET_without_Nan", data[mask_nan],dims =["cellsxtime_id"])
        pm.Data("FRET_0",data[:,0], dims = "cells")
    
        # =======================================================
        # 1. Population-level hyperpriors
        # =======================================================
        # phenotype-level parameters hyperpriors
        μ_beta0  = pm.LogNormal("μ_beta0",  0.005936848672639063, 0.005904786527505261)
        μ_beta1 = pm.LogNormal("μ_beta1", 0.047268093692721405/775, 0.07918652978955704)
        μ_beta2  = pm.Normal("μ_beta2",  -304.2607303288227/775, 540.2875623528233)
        μ_tau_discend  = pm.Normal("μ_tau_discend",  464.2787721210148, 81.68349775867487)
        
        
        # Note: how to get sigma for half Normal distribution from mean and variance from ifac data: 
        
        # For the LogNormal entries I used the standard formulas for a lognormal with underlying Normal(m, s):
        # variance = (e^{s^2} − 1) e^{2m + s^2}.
        
        # For Normals variance = sd².
        
        # For a half-normal: Var = σ²(1 − 2/π), so σ_half = sqrt(Var / (1 − 2/π)).
        σ_beta0  = pm.HalfNormal("σ_beta0",  0.009854)
        σ_beta1 = pm.HalfNormal("σ_beta1", 0.138370)
        σ_beta2  = pm.HalfNormal("σ_beta2", 896.2813)
        σ_tau_discend  = pm.HalfNormal("σ_tau_discend", 135.5045)
        
        # clone parameters  hyperpriors
        fret0_all_trim = trim_outliers(pd.Series(data[:,0]), q=0.10)
        
        μ_FRET0  = pm.Normal("μ_FRET0",  fret0_all_trim.mean(), fret0_all_trim.var())
        σ_FRET0  = pm.HalfNormal("σ_FRET0",  8.043285387793391e-05)
       
        
        
        # drug dose-level parameter hyperpriors
        if beta0_drug_dependent:
            μ_beta0_dd  = pm.LogNormal("μ_beta0_dd",  0.2, 0.1)
            σ_beta0_dd  = pm.HalfNormal("σ_beta0_dd",  0.05)
            
        # expectation
        μ_beta1_dd  = pm.LogNormal("μ_beta1_dd",  fret0_all_trim.mean()*0.1840243945712931, 1.0)
        μ_beta2_dd = pm.LogNormal("μ_beta2_dd", -fret0_all_trim.mean()*0.1840243945712931/(0.005936848672639063**2), 1.0)
        μ_tau_discend_dd  = pm.Normal("μ_tau_discend_dd",  1, 100)

        #variance
        σ_beta1_dd = pm.HalfNormal("σ_beta1_dd", 0.138370)
        σ_beta2_dd  = pm.HalfNormal("σ_beta2_dd", 896.2813)
        σ_tau_discend_dd  = pm.HalfNormal("σ_tau_discend_dd", 135.5045)
       
        # =======================================================
        # 2. Phenotype-level biological parameters
        # =======================================================
        beta0_ph  = pm.LogNormal("beta0_ph",  μ_beta0,  σ_beta0,  shape=N_pheno)
        beta1_ph = pm.LogNormal("beta1_ph", μ_beta1, σ_beta1, shape=N_pheno)
        beta2_ph  = pm.Normal("beta2_ph",  μ_beta2,  σ_beta2,  shape=N_pheno)
        tau_discend_ph  = pm.Normal("tau_discend_ph",  μ_tau_discend,  σ_tau_discend,  shape=N_pheno)

        # =======================================================
        # 3. Clone-level parameters
        # =======================================================
        FRET0_cl  = pm.Normal("FRET0_cl",  μ_FRET0,  σ_FRET0,  shape=N_clone)
        
        # =======================================================
        # 4. Drug-level 
        # =======================================================
        if beta0_drug_dependent:
            beta0_dd  = pm.LogNormal("beta0_dd",  μ_beta0_dd,  σ_beta0_dd,  shape=N_drug)
        beta1_dd = pm.LogNormal("beta1_dd", μ_beta1_dd, σ_beta1_dd, shape=N_drug)
        beta2_dd  = pm.Normal("beta2_dd",  μ_beta2_dd,  σ_beta2_dd,  shape=N_drug)
        tau_discend_dd  = pm.LogNormal("tau_discend_dd",  μ_tau_discend_dd, 
                                       σ_tau_discend_dd,  shape=N_drug)

        # =======================================================
        # 5. Weak cell-level offsets
        # =======================================================
        σ_offset = pm.HalfNormal("σ_offset", 0.01)

        δbeta0  = pm.Normal("δbeta0",  0, σ_offset, shape=N_cells)
        δbeta1 = pm.Normal("δbeta1", 0, σ_offset, shape=N_cells)
        δbeta2  = pm.Normal("δbeta2",  0, σ_offset, shape=N_cells)
        δtau_discend  = pm.Normal("δtau_discend",  0, σ_offset, shape=N_cells)

        # Combine drug_dose + phenotype + offsets
        dose_i = pt.take(drug_dose, drug_id)
        emax_i = pt.take(e_max, drug_id)
        ic50_i = pt.take(ic50, drug_id)
        
        
        emax_fct = 1 - (dose_i * emax_i) / (dose_i + ic50_i)
        if beta0_drug_dependent:
            beta0_cell  = beta0_dd[drug_id] * emax_fct  \
                + beta0_ph[phenotype_id]  + δbeta0
        else:
            beta0_cell  = beta0_ph[phenotype_id]  + δbeta0
        beta1_cell = beta1_dd[drug_id]* emax_fct + beta1_ph[phenotype_id] + δbeta1
        beta2_cell  = beta2_dd[drug_id] * emax_fct + beta2_ph[phenotype_id]  + δbeta2
        tau_discend_cell  = tau_discend_dd[drug_id] * emax_fct \
            + tau_discend_ph[phenotype_id]  + δtau_discend

        # Clone mappings
        FRET0_cell  = FRET0_cl[clone_id]
        
      
    

       

        # =======================================================
        # 6. Solve EAIM per cell
        # =======================================================
        yhat_list = []
        
        

        for i in range(N_cells):
            data_cell = data[i,:]
            mask_cell = ~np.isnan(data_cell)
            time_points_without_nan_cell = time_matrix[i,mask_cell]
            print(beta0_cell[i].eval(), 
            beta1_cell[i].eval(), 
            beta2_cell[i].eval(),
            tau_discend_cell[i].eval(),
            FRET0_cell[i].eval(),
            time_matrix[i,mask_cell])
            sol = ap_models.FRETexp_bayesian(time_points_without_nan_cell,
                                             beta0_cell[i], 
                                             beta1_cell[i], 
                                             beta2_cell[i],
                                             tau_discend_cell[i],
                                             FRET0_cell[i])
            print( sol.eval())
            yhat_list.append(sol)

        # y_hat = pt.stack(yhat_list)
        y_hat = pt.concatenate(yhat_list)
        print(yhat_list,len(data[mask_nan]))

        # =======================================================
        # 7. Likelihood (NaNs are ignored automatically)
        # =======================================================
        σ = pm.HalfNormal("σ", 0.1)
        pm.Normal("obs", mu=y_hat, sigma=σ, observed=data[mask_nan])

    return model

def parse_arguments_partial_pooling():
   
    parser = argparse.ArgumentParser(
        description='Run bayesian hierarchical apoptosis model inference')
    parser.add_argument('--config_file', default = "../config_parameters.ini", help='Path to configuration files')
  

    return parser.parse_args()



# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    args = parse_arguments_partial_pooling()
    config_pipeline_file = args.config_file
    
    # Open config file
    config = configparser.ConfigParser()
    # print(config_file)
    config.read(config_pipeline_file)
    # default_folder = os.path.join('/mnt', 'nas', '02_Analyzed_data', 'Image_analysis', '2025')
    data_file = config.get('PathData','path_mp_laptop')
   
    
    
    #load data
    df_fret = upload_data.load_data_and_phenotype(path_to_pickle = data_file,
                                                  config = config,
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
    
   
    
    # # Build Model
    # model = build_hierarchical_model_FRETexp(
    #     time_points=time_points,
    #     data=data,
    #     clone_id=clone_id,
    #     phenotype_id=phenotype_id,
    #     drug_dose = drug_dose,
    #     e_max = e_max, 
    #     ic50=ic50,  
    #     drug_id = drug_id, 
    #     beta0_drug_dependent = True

    # )
    
    model = build_hierarchical_model_FRETexp(
        time_points=time_points,
        data=data,
        clone_id=clone_id,
        phenotype_id=phenotype_id,
        drug_dose = drug_dose,
        e_max = e_max, 
        ic50=ic50,  
        drug_id = drug_id, 
        beta0_drug_dependent = True

    )
    
    # Run inference
    # with model:
    #     trace = pm.sample(tune=500, draw=1000, cores=4, chains=4)
        
    summary = az.summary(trace, hdi_prob=0.95)
    print(summary)
    
  
    
   