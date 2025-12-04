#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 10:11:41 2025

@author: mpere
"""
# =============================================================================
# IMPORT
# =============================================================================
import os
import pandas as pd
import plotly
import pickle
import numpy as np
import matplotlib.pyplot as plt
import configparser
import argparse
from datetime import date
# =============================================================================
# FUNCTIONS
# =============================================================================
def ensure_directory_exists(data_pathway):
    # Check if the directory exists
    if not os.path.exists(data_pathway):
        # Directory does not exist, so create it
        os.makedirs(data_pathway)
        print(f"Directory '{data_pathway}' created.")
    else:
        # Directory exists
        print(f"Directory '{data_pathway}' already exists.")
        
        
def add_cell_phenotype_6h(df_fret):
    df_fret["phenotype_6h"] = np.where(df_fret["Death Time"] <= 360, "S", "T")
    return df_fret


    
    
def print_database_summary_per_cell_line(df_fret):
    # Count number of S and T per cell line and dose
    summary = (
        df_fret
        .groupby(['Dose', 'Cell Line', 'phenotype_6h'])
        .size()
        .unstack(fill_value=0)  # Separate S/T into columns
        .reset_index()
    )

    # Combine counts into "S:x | T:y" or "-" if both are zero
    def format_counts(row):
        s = row.get('S', 0)
        t = row.get('T', 0)
        return f"S:{s} | T:{t}" if (s > 0 or t > 0) else "-"

    summary['Summary'] = summary.apply(format_counts, axis=1)

    # Pivot so rows = Dose, columns = Cell Line
    table = summary.pivot(index='Dose', columns='Cell Line', values='Summary').fillna('-')

      # --- Add totals row per S and T ---
    totals = df_fret.groupby(['Cell Line', 'phenotype_6h']).size().unstack(fill_value=0)
    totals_row = {cl: f"S:{totals.get('S', {}).get(cl, 0)} | T:{totals.get('T', {}).get(cl, 0)}"
                  for cl in table.columns}
    table.loc['Total'] = totals_row

    # Print the table nicely
    print("\n=== DATABASE SUMMARY ===")
    print(table.to_string())
    print()
    
def print_database_summary_per_clone(df_fret):
    """
    Print a database summary of counts of sensitive (S) and tolerant (T) cells
    grouped by Dose, Cell Line, phenotype_6h, and Clone.
    """

    # Count number of S and T per cell line, dose, and clone
    summary = (
        df_fret
        .groupby(['Dose', 'Cell Line', 'Clone', 'phenotype_6h'])
        .size()
        .unstack(fill_value=0)  # Separate S/T into columns
        .reset_index()
    )

    # Combine counts into "S:x | T:y" or "-" if both are zero
    def format_counts(row):
        s = row.get('S', 0)
        t = row.get('T', 0)
        return f"S:{s} | T:{t}" if (s > 0 or t > 0) else "-"

    summary['Summary'] = summary.apply(format_counts, axis=1)

    # Pivot so rows = Dose, columns = (Cell Line, Clone)
    table = summary.pivot(index='Dose', columns=['Cell Line', 'Clone'], values='Summary').fillna('-')

    # --- Add totals row per S and T ---
    totals = df_fret.groupby(['Cell Line', 'Clone', 'phenotype_6h']).size().unstack(fill_value=0)
    totals_row = {}
    for cl, clone in table.columns:
        s_total = totals.get('S', {}).get((cl, clone), 0)
        t_total = totals.get('T', {}).get((cl, clone), 0)
        totals_row[(cl, clone)] = f"S:{s_total} | T:{t_total}"

    table.loc['Total'] = totals_row

    # --- Print the table nicely ---
    print("\n=== DATABASE SUMMARY ===")
    print(table.to_string())
    print()


def print_database_summary_per_clone_sensitive_percentage(df_fret):
    """
    Print a database summary of counts of sensitive (S) and tolerant (T) cells
    grouped by Dose, Cell Line, phenotype_6h, and Clone.
    """

    # Count number of S and T per cell line, dose, and clone
    summary = (
        df_fret
        .groupby(['Dose', 'Cell Line', 'Clone', 'phenotype_6h'])
        .size()
        .unstack(fill_value=0)  # Separate S/T into columns
        .reset_index()
    )
    
   

    # Combine counts into "S:x | T:y" or "-" if both are zero
    def format_counts(row):
        s = row.get('S', 0)
        t = row.get('T', 0)
        return f"{int(100*s/(s+t))}% ({s+t})" if (s > 0 or t > 0) else "-"

    summary['Summary'] = summary.apply(format_counts, axis=1)

    # Pivot so rows = Dose, columns = (Cell Line, Clone)
    table = summary.pivot(index='Dose', columns=['Cell Line', 'Clone'], values='Summary').fillna('-')

    # --- Add totals row per S and T ---
    totals = df_fret.groupby(['Cell Line', 'Clone', 'phenotype_6h']).size().unstack(fill_value=0)
    totals_row = {}
    for cl, clone in table.columns:
        s_total = totals.get('S', {}).get((cl, clone), 0)
        t_total = totals.get('T', {}).get((cl, clone), 0)
        totals_row[(cl, clone)] = f"{int(100*s_total/(s_total+t_total))}% ({s_total+t_total})"

    table.loc['Total'] = totals_row

    # --- Print the table nicely ---
    print("\n=== DATABASE SUMMARY ===")
    table.index = [int(i)  if i !='Total' else np.nan for i in table.index]
    table.sort_index(inplace=True)
    print(table.to_string())
    print()

def load_data_and_phenotype(path_to_pickle,config, include_HPAF = False):
    
    
    # Load data
    df_fret = pd.read_pickle(path_to_pickle)
    
    print('Reading ', path_to_pickle)
    
    # remove discarded cells
    df_fret = df_fret[~np.isnan(df_fret['Death Time'])]
    
    # remove Hela clones other than JR2
    df_fret = df_fret[~df_fret.Clone.isin(['S1', 'L', 'FLIP S1', 'FLIP L'])]
    

    
    # rename K710% by KC7 (bad naming during LCM)
    df_fret['Clone'] = df_fret['Clone'].replace({'KC710%': 'KC7'})
    df_fret["IC50"] = df_fret["Cell Line"].map({
    "PANC": config.getfloat('IC50','IC50_PANC'),
    "HeLa": config.getfloat('IC50','IC50_HeLa'),
    "DLD": config.getfloat('IC50','IC50_DLD'),
    "SW837": config.getfloat('IC50','IC50_SW837')
    })
    
    df_fret["Emax"] = df_fret["Cell Line"].map({
    "PANC": config.getfloat('Emax','Emax_PANC'),
    "HeLa": config.getfloat('Emax','Emax_HeLa'),
    "DLD": config.getfloat('Emax','Emax_DLD'),
    "SW837": config.getfloat('Emax','Emax_SW837')
    })
    
    # remove 2023 12 06 - cf One Note
    # pbm KC4 and KC7: bad manip
    d0 = date(2023, 12, 6)
    
    # pbm KC7 and KC8
    d1 = date(2024,11,13)
    d2 = date(2024,11,14)
    d3 = date(2024,11,19)
    d4 = date(2024,11,20)
    d5 = date(2024,11,21)
    
    d6 = date(2023,12,21)
    d7 = date(2024,10,15)
    
    # pbm HeLa
    d8 = date(2023,10,17)
    d9 = date(2023,10,19)
    
    # df_weird = pd.DataFrame()
    for d in [d0, d1,d2,d3,d4,d5,d6,d7,d8,d9]:
        # df_weird = pd.concat((df_weird,df_fret[df_fret.Date == d] ), axis = 0)
        df_fret = df_fret[df_fret.Date != d]
       
    
    # determine cell phenotype
    df_fret = add_cell_phenotype_6h(df_fret)
    # df_weird = add_cell_phenotype_6h(df_weird)
 
    
    # print database summary
    print_database_summary_per_clone(df_fret)
    print_database_summary_per_clone_sensitive_percentage(df_fret)
    
    if include_HPAF:
        return df_fret
    else:
        return df_fret[df_fret['Cell Line']!= 'HPAFII']
 
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
    # default_folder = os.path.join('/mnt', 'nas', '02_Analyzed_data', 'Image_analysis', '2025')
    data_file = config.get('PathData','path_dataset_mmg_cluster')
    data_file = config.get('PathData','path_mp_laptop')

    df_fret = load_data_and_phenotype(data_file,config,
                                     include_HPAF = False)
    