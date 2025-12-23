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
import sys

from icecream import ic
from pathlib import Path    
from IPython.display import display



# Add parent folder to Python path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

import upload_data 
import apoptosis_models as ap_models
import FRETexp_parameter_distribution_analysis as prior_fit



# =============================================================================
# FUNCTIONS
# =============================================================================

def trim_outliers(series, q=0.10):
    """Remove the lowest and highest q fraction of data."""
    low = series.quantile(q)
    high = series.quantile(1 - q)
    return series[(series >= low) & (series <= high)]


def return_feature_value_according_to_level_name(level_name,df_fret):
    if level_name == 'homogeneous':
        return 'homogeneous'
    else:
        if level_name =='cellline':
            feature = 'Cell Line'
        if level_name =='clone':
            feature = 'Clone'
        if level_name =='drug':
            feature = 'clone_drug'
        if level_name == 'phenotype':
            feature = 'clone_drug_phenotype'
        return df_fret[feature].drop_duplicates().to_numpy()


def generate_prior_dicts(FOSBE2022_data_path, parameter, level_name, level_id,  df_fret):
        mean_prior, var_prior = prior_fit.get_prior_distribution_Hela(FOSBE2022_data_path, parameter)
        if level_name == 'cellline' or level_name == 'clone' or level_name == 'homogeneous':
            nb_cat = len(np.unique(level_id))        
            return [mean_prior[level_name]]*nb_cat, [var_prior[level_name]]*nb_cat
        elif level_name == 'drug':
            nb_cat =  len(np.unique(level_id))  
            mean_list = []
            var_list = []
            df_unique = (
                df_fret.assign(level_id=level_id)
                .drop_duplicates(subset="level_id")
                .drop(columns="level_id")
            )
            print(df_unique)
            for  id_r, row in df_unique.iterrows():
                if row['relative_dose'] <0.3:
                    mean_list.append(mean_prior['drug_low'])
                    var_list.append(var_prior['drug_low'])
                elif row['relative_dose'] >0.7:
                    mean_list.append(mean_prior['drug_high'])
                    var_list.append(var_prior['drug_high'])
                else:
                    mean_list.append(mean_prior['drug_ic50'])
                    var_list.append(var_prior['drug_ic50'])
            return mean_list, var_list
        else:
            print('Phenotype level')
            nb_cat =  len(np.unique(level_id))  
            mean_list = []
            var_list = []
            df_unique = (
                df_fret.assign(level_id=level_id)
                .drop_duplicates(subset="level_id")
                .drop(columns="level_id")
            )
            
            for id_r,row in df_unique.iterrows():
                if row['relative_dose'] <0.3:
                    if row['phenotype'] == 'S':
                        mean_list.append(mean_prior['drug_low_sen'])
                        var_list.append(var_prior['drug_low_sen'])
                    else:
                        mean_list.append(mean_prior['drug_low_tol'])
                        var_list.append(var_prior['drug_low_tol'])
                elif row['relative_dose'] >0.7:
                    if row['phenotype'] == 'S':
                        mean_list.append(mean_prior['drug_high_sen'])
                        var_list.append(var_prior['drug_high_sen'])
                    else:
                        mean_list.append(mean_prior['drug_high_tol'])
                        var_list.append(var_prior['drug_high_tol'])
                else:
                    if row['phenotype'] == 'S':
                        mean_list.append(mean_prior['drug_ic50_sen'])
                        var_list.append(var_prior['drug_ic50_sen'])
                    else:
                        mean_list.append(mean_prior['drug_ic50_tol'])
                        var_list.append(var_prior['drug_ic50_tol'])
                 
            return mean_list, var_list
        


def build_pooled_model_FRETexp(
    time_points, data, 
    level_id ,
    parameter_levels,
    FOSBE2022_data_path
):
    N_cells, T_len = data.shape
    N_pheno = len(np.unique(phenotype_id))
    N_clone = len(np.unique(clone_id))
    N_clone = len(np.unique(cellline_id))
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
        # 1. define each parameter distribution according to corresponding pop. level
        # =======================================================
        # beta0
        beta0_level_name = parameter_levels['beta0']
        beta0_level_id = level_id[beta0_level_name]
        mean_beta0_cellline, var_beta0_cellline = generate_prior_dicts(FOSBE2022_data_path,
                                                                        'beta0', 
                                                                        beta0_level_name, 
                                                                        beta0_level_id,
                                                                        df_fret)
        print('mean_beta0_cellline',mean_beta0_cellline)
        # create prior distribution for each category of the subpopulation
        nb_cat = len(np.unique(beta0_level_id))
        dict_distribution_beta0 = {}
        categories_beta0 = return_feature_value_according_to_level_name(beta0_level_name,df_fret)
        for cat_id, cat in enumerate(np.unique(beta0_level_id)):
            print('cat_id, cat',cat_id, cat)
            dict_distribution_beta0[cat] = pm.LogNormal(f"beta0_{categories_beta0[cat]}",  np.log(mean_beta0_cellline[cat_id]),  var_beta0_cellline[cat_id])

        # beta1
        beta1_level_name = parameter_levels['beta1']
        beta1_level_id = level_id[beta1_level_name]
        mean_beta1_cellline, var_beta1_cellline = generate_prior_dicts(FOSBE2022_data_path,
                                                                        'beta1', 
                                                                        beta1_level_name, 
                                                                        beta1_level_id,
                                                                        df_fret)
        nb_cat = len(np.unique(beta1_level_id))
        dict_distribution_beta1 = {}
        categories_beta1 = return_feature_value_according_to_level_name(beta1_level_name,df_fret)
        for cat_id, cat in enumerate(np.unique(beta1_level_id)):
            print('cat_id, cat',cat_id, cat)
            dict_distribution_beta1[cat] = pm.LogNormal(f"beta1_{categories_beta1[cat]}",  np.log(mean_beta1_cellline[cat_id]),  var_beta1_cellline[cat_id])

        # beta2
        beta2_level_name = parameter_levels['beta2']
        beta2_level_id = level_id[beta2_level_name]
        mean_beta2_cellline, var_beta2_cellline = generate_prior_dicts(FOSBE2022_data_path,
                                                                        'beta2', 
                                                                        beta2_level_name, 
                                                                        beta2_level_id,
                                                                        df_fret)
        nb_cat = len(np.unique(beta2_level_id))
        dict_distribution_beta2 = {}
        categories_beta2 = return_feature_value_according_to_level_name(beta2_level_name,df_fret)
        for cat_id, cat in enumerate(np.unique(beta2_level_id)):
            print('cat_id, cat',cat_id, cat)
            dict_distribution_beta2[cat] = pm.Normal(f"beta2_{categories_beta2[cat]}",  mean_beta2_cellline[cat_id],  var_beta2_cellline[cat_id])

        # tau disc end
        tau_discend_level_name = parameter_levels['tau_discend']
        tau_discend_level_id = level_id[tau_discend_level_name]
        mean_tau_discend_cellline, var_tau_discend_cellline = generate_prior_dicts(FOSBE2022_data_path,
                                                                        'tau_DISCend_first_estimate', 
                                                                        tau_discend_level_name, 
                                                                        tau_discend_level_id,
                                                                        df_fret)
        nb_cat = len(np.unique(tau_discend_level_id))
        dict_distribution_tau_discend = {}
        categories_tau_discend = return_feature_value_according_to_level_name(tau_discend_level_name,df_fret)
        for cat_id, cat in enumerate(np.unique(tau_discend_level_id)):
            print('cat_id, cat',cat_id, cat)
            dict_distribution_tau_discend[cat] = pm.Normal(f"tau_discend_{categories_tau_discend[cat]}",  mean_tau_discend_cellline[cat_id],  var_tau_discend_cellline[cat_id])

        # FRET0
        FRET0_level_name = parameter_levels['FRET0']
        FRET0_level_id = level_id[FRET0_level_name]
        mean_FRET0_cellline, var_FRET0_cellline = generate_prior_dicts(FOSBE2022_data_path,
                                                                        'T=5min', 
                                                                        FRET0_level_name, 
                                                                        FRET0_level_id,
                                                                        df_fret)
        nb_cat = len(np.unique(FRET0_level_id))
        dict_distribution_FRET0 = {}
        categories_FRET0 = return_feature_value_according_to_level_name(FRET0_level_name,df_fret)
        for cat_id, cat in enumerate(np.unique(FRET0_level_id)):
            print('cat_id, cat',cat_id, cat)
            dict_distribution_FRET0[cat] = pm.Normal(f"FRET0_{categories_FRET0[cat]}",  mean_FRET0_cellline[cat_id],  var_FRET0_cellline[cat_id])

      

       

        # =======================================================
        # 6. Compute FRETexp per cell
        # =======================================================
        yhat_list = []
        
        

        for i in range(N_cells):
            data_cell = data[i,:]
            cat_beta0 = beta0_level_id[i]
            cat_beta1 = beta1_level_id[i]
            cat_beta2 = beta2_level_id[i]
            cat_tau_discend = tau_discend_level_id[i]
            cat_FRET0 = FRET0_level_id[i]
            mask_cell = ~np.isnan(data_cell)
            time_points_without_nan_cell = time_matrix[i,mask_cell]
            print(df_fret['clone_drug_phenotype'].iloc[i],cat_beta0 ,beta0_level_id[i],dict_distribution_beta0[cat_beta0].eval())
            print(df_fret['clone_drug_phenotype'].iloc[i],cat_beta1 ,beta1_level_id[i],dict_distribution_beta1[cat_beta1].eval())
            print(df_fret['clone_drug_phenotype'].iloc[i],cat_beta2 ,beta2_level_id[i],dict_distribution_beta2[cat_beta2].eval())
            print(df_fret['clone_drug_phenotype'].iloc[i],cat_FRET0 ,FRET0_level_id[i],dict_distribution_FRET0[cat_FRET0].eval())
            print(df_fret['clone_drug_phenotype'].iloc[i],cat_tau_discend ,tau_discend_level_id[i],dict_distribution_tau_discend[cat_tau_discend].eval())
            print()
            # beta1_cell.eval(), 
            # beta2_cell.eval(),
            # tau_discend_cell.eval(),
            # FRET0_cell.eval(),
            # time_matrix[i,mask_cell])
            sol = ap_models.FRETexp_bayesian(time_points_without_nan_cell,
                                             dict_distribution_beta0[cat_beta0], 
                                             dict_distribution_beta1[cat_beta1], 
                                             dict_distribution_beta2[cat_beta2],
                                             dict_distribution_tau_discend[cat_tau_discend],
                                             dict_distribution_FRET0[cat_FRET0])
            # print( sol.eval())
            yhat_list.append(sol)

        # y_hat = pt.stack(yhat_list)
        y_hat = pt.concatenate(yhat_list)
        # print(len(y_hat.eval()),len(data[mask_nan]))

        # =======================================================
        # 7. Likelihood (NaNs are ignored automatically)
        # =======================================================
        σ = pm.HalfNormal("σ", 0.1)
        pm.Normal("obs", mu=y_hat, sigma=σ, observed=data[mask_nan])

    return model

def parse_arguments_partial_pooling():
   
    parser = argparse.ArgumentParser(
        description='Run bayesian hierarchical apoptosis model inference')
    parser.add_argument('--config_file', default = "config_parameters.ini", help='Path to configuration files')
  

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
    data_file = config.get('PathData','path_dataset')

    FOSBE2022_data_path = config.get('PathData','path_FOSBE2022')
    output_path = config.get('PathData','path_output')
    Path(output_path).mkdir(parents=True, exist_ok=True)   
   
    
    
    #load data
    df_fret = upload_data.load_data_and_phenotype(path_to_pickle = data_file,
                                                  config = config,
                                                  include_HPAF = False)
    
    
    df_fret = df_fret.iloc[0:100,:]

    print(df_fret['Dose'].drop_duplicates())
    print(list(df_fret.index))

    # define time
    time_cols = [c for c in df_fret.columns if isinstance(c, (int, float))]
    time_points = np.array(time_cols)
    
    
    # define drug dose
    drug_dose = np.array(df_fret['Dose'], dtype = float)
    ic50 = np.array(df_fret['IC50'])

    relative_dose = drug_dose /(drug_dose + ic50)
    df_fret['relative_dose'] = relative_dose
    print('relative dose',relative_dose)
    print(df_fret['phenotype'])
    
    # define categories inside each population level
    clone_labels = df_fret["Clone"].astype("category")
    clone_id = clone_labels.cat.codes.values
    
    cell_line_labels = df_fret["Cell Line"].astype("category")
    cellline_id = cell_line_labels.cat.codes.values
    
    df_fret['clone_drug_phenotype'] = [f"{cl}_{dd}_{ph}" for cl,dd,ph in zip(df_fret['Clone'],df_fret['Dose'],df_fret['phenotype'])]
    df_fret['cellline_drug_phenotype'] = [f"{cl}_{dd}_{ph}" for cl,dd,ph in zip(df_fret['Cell Line'],df_fret['Dose'],df_fret['phenotype'])]
    
    df_fret['clone_drug'] = [f"{cl}_{dd}" for cl,dd in zip(df_fret['Clone'],df_fret['Dose'])]
    df_fret['cellline_drug'] = [f"{cl}_{dd}" for cl,dd in zip(df_fret['Cell Line'],df_fret['Dose'])]
    
    
    phenotype_labels = df_fret["clone_drug_phenotype"].astype("category")
    phenotype_id = phenotype_labels.cat.codes.values
    
    drug_labels = df_fret["clone_drug"].astype("category")
    drug_id = drug_labels.cat.codes.values
    
    homogeneous_id = np.zeros(drug_id.shape,dtype=int)
    
    
    # extract into N_cells × T matrix
    data = df_fret[time_cols].to_numpy()

    

    mean_beta0_cellline, var_beta0_cellline = generate_prior_dicts(FOSBE2022_data_path, 'beta0', 'drug', drug_id,df_fret)



    model = build_pooled_model_FRETexp(
        time_points=time_points,
        data=data,
        level_id =  {'cellline':cellline_id,  'clone': clone_id,'drug': drug_id, 'phenotype':phenotype_id, 'homogeneous':homogeneous_id},
        parameter_levels = {'beta0':'phenotype','beta1':'drug','beta2':'clone','tau_discend':'cellline','FRET0':'homogeneous'},
        FOSBE2022_data_path = FOSBE2022_data_path

    )
    print(phenotype_id)
    # # Run inference
    with model:
        trace = pm.sample(tune=1000, draw=1000, cores=4, chains=4)
    # %%  
    summary = az.summary(trace, hdi_prob=0.95)
    az.to_netcdf(trace, os.path.join(output_path,"trace.nc"))
    print(summary)

 
   

    pm.model_to_graphviz(model)
    graph = pm.model_to_graphviz(model)
    graph.render(os.path.join(output_path,"model_graph"), format="png", cleanup=True)

    
  
    
   