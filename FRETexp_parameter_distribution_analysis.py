#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 14:56:29 2025

@author: mpere
"""
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import configparser
import os
# =============================================================================
# MAIN
# =============================================================================
def trim_outliers(series, q=0.10):
    """Remove the lowest and highest q fraction of data."""
    low = series.quantile(q)
    high = series.quantile(1 - q)
    return series[(series >= low) & (series <= high)]


def get_prior_distribution_Hela(data_path, parameter = 'beta0'):
    """
    Docstring for get_prior_distribution_Hela
    
    :param data_path: absolute path to FSBE 2022 parameters
    :param parameter: str, parameters name in FOSBE 2022 parmaeters dataframes: beta0, beta1, beta2, tau_DISCend_first_estimate,T=5min
    """
     # load data
    ifac_t5 = pd.read_csv(os.path.join(data_path,"FOSBE2022_parameters/2015_ROUX_testHeLafinalRest_T005_600min_CUT.csv"), 
                          sep = '\t',index_col = 0)
    ifac_t10 = pd.read_csv(os.path.join(data_path,"FOSBE2022_parameters/2015_ROUX_testHeLafinalRest_T010_600min_CUT.csv"), 
                           sep = '\t', index_col = 0)
    ifac_t25 = pd.read_csv(os.path.join(data_path,"FOSBE2022_parameters/2015_ROUX_testHeLafinalRest_T025_600min_CUT.csv"),
                           sep = '\t', index_col = 0)
    ifac_t50 = pd.read_csv(os.path.join(data_path,"FOSBE2022_parameters/2015_ROUX_testHeLafinalRest_T050_600min_CUT.csv"), 
                           sep = '\t', index_col = 0)
    
    all_concat = pd.concat([ifac_t5,ifac_t10,ifac_t25,ifac_t50], axis = 0)
    
    low = pd.concat([ifac_t5,ifac_t10])

    ic50 = ifac_t25

    high = ifac_t50

    low_tol = low[low.phenotype == 1.0]
    low_sen = low[low.phenotype == 0.0]

    ic50_tol = ic50[ic50.phenotype == 1.0]
    ic50_sen = ic50[ic50.phenotype == 0.0]

    high_tol = high[high.phenotype == 1.0]
    high_sen = high[high.phenotype == 0.0]


    mean = {'homogeneous':trim_outliers(all_concat[parameter]).mean(),
            'cellline':trim_outliers(all_concat[parameter]).mean(),
            'clone': trim_outliers(all_concat[parameter]).mean(),
            'drug_low':trim_outliers(low[parameter]).mean(),
            'drug_ic50':trim_outliers(ic50[parameter]).mean(),
            'drug_high':trim_outliers(high[parameter]).mean(),
            'drug_low_tol':trim_outliers(low_tol[parameter]).mean(),
            'drug_ic50_tol':trim_outliers(ic50_tol[parameter]).mean(),
            'drug_high_tol':trim_outliers(high_tol[parameter]).mean(),
            'drug_low_sen':trim_outliers(low_sen[parameter]).mean(),
            'drug_ic50_sen':trim_outliers(ic50_sen[parameter]).mean(),
            'drug_high_sen':trim_outliers(high_sen[parameter]).mean()}
    var = {'homogeneous':trim_outliers(all_concat[parameter]).var(),
            'cellline':trim_outliers(all_concat[parameter]).var(),
            'clone': trim_outliers(all_concat[parameter]).var(),
            'drug_low':trim_outliers(low[parameter]).var(),
            'drug_ic50':trim_outliers(ic50[parameter]).var(),
            'drug_high':trim_outliers(high[parameter]).var(),
            'drug_low_tol':trim_outliers(low_tol[parameter]).var(),
            'drug_ic50_tol':trim_outliers(ic50_tol[parameter]).var(),
            'drug_high_tol':trim_outliers(high_tol[parameter]).var(),
            'drug_low_sen':trim_outliers(low_sen[parameter]).var(),
            'drug_ic50_sen':trim_outliers(ic50_sen[parameter]).var(),
            'drug_high_sen':trim_outliers(high_sen[parameter]).var()}
    return mean, var

def parse_arguments_upload_data():
   
    parser = argparse.ArgumentParser(
        description='Run bayesian hierarchical apoptosis model inference')
    parser.add_argument('--config_file', default = "config_parameters.ini", help='Path to configuration files')
  

    return parser.parse_args()



# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    args = parse_arguments_upload_data()
    config_pipeline_file = args.config_file
    
    # Open config file
    config = configparser.ConfigParser()
    # print(config_file)
    config.read(config_pipeline_file)
   
    data_path = config.get('PathData','path_FOSBE2022')
    print(data_path )
    mean, var = get_prior_distribution_Hela(data_path)

    for (mk,mv),(vk,vv) in zip(mean.items(),var.items()):
        print(f'Mean: {mk} = {mv}\nVar: {vk} = {vv}')

    # load data
    ifac_t5 = pd.read_csv("/home/marielle/Bureau/00_Projects/02_bayesian_apoptosis/data2025/FOSBE2022_parameters/2015_ROUX_testHeLafinalRest_T005_600min_CUT.csv", 
                          sep = '\t',index_col = 0)
    ifac_t10 = pd.read_csv("/home/marielle/Bureau/00_Projects/02_bayesian_apoptosis/data2025/FOSBE2022_parameters/2015_ROUX_testHeLafinalRest_T010_600min_CUT.csv", 
                           sep = '\t', index_col = 0)
    ifac_t25 = pd.read_csv("/home/marielle/Bureau/00_Projects/02_bayesian_apoptosis/data2025/FOSBE2022_parameters/2015_ROUX_testHeLafinalRest_T025_600min_CUT.csv",
                           sep = '\t', index_col = 0)
    ifac_t50 = pd.read_csv("/home/marielle/Bureau/00_Projects/02_bayesian_apoptosis/data2025/FOSBE2022_parameters/2015_ROUX_testHeLafinalRest_T050_600min_CUT.csv", 
                           sep = '\t', index_col = 0)
    





    # decomment to see per phenotype : Sensitive = 0, Tolerant = 1
    # phenotype = 0.0
    
    
    # ifac_t5 = ifac_t5[ifac_t5.phenotype == phenotype]
    # ifac_t10 = ifac_t10[ifac_t10.phenotype == phenotype]
    # ifac_t25 = ifac_t25[ifac_t25.phenotype == phenotype]
    # ifac_t50 = ifac_t50[ifac_t50.phenotype == phenotype]
    
    
    # plot layout
    
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Raw data')
    axs[0,0].set_title(r"$\beta_0$")
    axs[0,1].set_title(r"$\beta_1$")
    axs[1,0].set_title(r"$\beta_2$")
    axs[1,1].set_title(r"$\tau_{DISC.end}$")
    
    
    for (data,lab) in zip([ifac_t5,ifac_t10,ifac_t25,ifac_t50],['T05','T10','T25','T50']):
    
        axs[0,0].hist(data['beta0'], label = lab)
        axs[0,1].hist(data['beta1'], label = lab)
        axs[1,0].hist(data['beta2'], label = lab)
        axs[1,1].hist(data['tau_DISCend_first_estimate'], label = lab)
    
    
    for ax in axs.flat:
        ax.legend()
        
    fig, axs = plt.subplots(3,2)
    fig.suptitle('Trim data')
    axs[0,0].set_title(r"$\beta_0$")
    axs[0,1].set_title(r"$\beta_1$")
    axs[1,0].set_title(r"$\beta_2$")
    axs[1,1].set_title(r"$\tau_{DISC.end}$")
    axs[2,0].set_title('FRET_0')
    axs[2,1].set_title(r'$\alpha_0$')
    
   
    
    
    beta0_t_total = pd.Series()
    beta1_t_total = pd.Series()
    beta2_t_total = pd.Series()
    tau_t_total = pd.Series()
    fret0_t_total = pd.Series()
    alpha0_t_total = pd.Series()
    
    
    for (data,lab) in zip([ifac_t5,ifac_t10,ifac_t25,ifac_t50],['T05','T10','T25','T50']):
    
        axs[0,0].hist(trim_outliers(data['beta0']), label=lab)
        axs[0,1].hist(trim_outliers(data['beta1']), label=lab)
        axs[1,0].hist(trim_outliers(data['beta2']), label=lab)
        axs[1,1].hist(trim_outliers(data['tau_DISCend_first_estimate']), label=lab)
        axs[2,0].hist(trim_outliers(data['T=5min']), label=lab)
        axs[2,1].hist(trim_outliers(data['alpha0']), label=lab)
        
        # Trimmed series
        beta0_t = trim_outliers(data['beta0'])
        beta1_t = trim_outliers(data['beta1'])
        beta2_t = trim_outliers(data['beta2'])
        tau_t   = trim_outliers(data['tau_DISCend_first_estimate'])
        fret0_t = trim_outliers(data['T=5min'])
        alpha0_t = trim_outliers(data['alpha0'])
        
        beta0_t_total = pd.concat([beta0_t_total,beta0_t])
        beta1_t_total = pd.concat([beta1_t_total,beta1_t])
        beta2_t_total = pd.concat([beta2_t_total,beta2_t])
        tau_t_total = pd.concat([tau_t_total,tau_t])
        fret0_t_total = pd.concat([fret0_t_total, fret0_t])
        alpha0_t_total = pd.concat([alpha0_t_total, alpha0_t])
                   
     

        
        
        # Print variances
        print(f"\n\nMean and Variance (trimmed) for {lab}:")
        print("beta0:", beta0_t.mean(), beta0_t.var())
        print("beta1:",  beta1_t.mean(),beta1_t.var())
        print("beta2:",  beta2_t.mean(),beta2_t.var())
        print("tau_DISCend_first_estimate:", tau_t.mean(),tau_t.var())
        print("fret0:", fret0_t.mean(), fret0_t.var())
        print("alpha0:", alpha0_t.mean(), alpha0_t.var())
            
    
    for ax in axs.flat:
        ax.legend()
        
    print("\n\nMean and Variance (trimmed) for all combined:")
    print("beta0:", beta0_t_total.mean(), beta0_t_total.var())
    print("beta1:",  beta1_t_total.mean(),beta1_t_total.var())
    print("beta2:",  beta2_t_total.mean(),beta2_t_total.var())
    print("tau_DISCend_first_estimate:", tau_t.mean(),tau_t.var())
    print("fret0:",  fret0_t_total.mean(),fret0_t_total.var())
    print("alpha0:",  alpha0_t_total.mean(),alpha0_t_total.var(),'\n\n')
    
    
    fig, axs = plt.subplots(2,2)
    axs[0,0].set_title(r"T05")
    axs[0,1].set_title(r"T10")
    axs[1,0].set_title(r"T25")
    axs[1,1].set_title(r"T50")
    
    for ax, data in zip(axs.flat,[ifac_t5,ifac_t10,ifac_t25,ifac_t50]):
        cols = [col  for col in data.columns if col.startswith('T=')]
        time_point = [int(col.replace('T=','').replace('min','')) for col in cols]
        s = data[data.phenotype == 0.0]
        t = data[data.phenotype == 1.0]
        t[cols].T.plot(ax = ax,color = 'b')
        s[cols].T.plot(ax = ax, color = 'r')
        
    
    
plt.show()  
    