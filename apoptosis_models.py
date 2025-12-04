#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025 11 10

@author: %(Marielle Péré)s
"""

# =============================================================================
# IMPORT
# =============================================================================
import numpy as np
import pytensor
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import scipy.io
from pymc.ode import DifferentialEquation

def C8_phenomenological_model(t,k,t0):
    return k*(t-t0)**2

def FRETexp(time_vector, beta0, beta1, beta2, tau_discend,FRET_0 = 0):
    """
    Calculate the FRET signal over time using a vectorized approach.

    Parameters:
    -----------
    time_vector : pytensor.tensor.variable.TensorVariable
        PyTensor tensor representing the time points.
    beta0, beta1, beta2 : pytensor.tensor.variable.TensorVariable
        PyTensor tensors representing model parameters that control the growth and decay rates.
    tau_discend : pytensor.tensor.variable.TensorVariable
        PyTensor tensor representing the time at which a transition occurs in the model.

    Returns:
    --------
    FRET : pytensor.tensor.variable.TensorVariable
        Computed FRET values for each time point in time_vector.
    """
    
    
    if time_vector.ndim == 1 :
        if type(tau_discend) == float: #tau is a number
            # Precompute constant values
            FRET_at_tau_discend = beta1 * tau_discend + beta2 * (1 - np.exp(-beta0 * tau_discend))
            post_tau_discend_constant = beta1 / beta0 + beta2 * np.exp(-beta0 * tau_discend)
           
            # Create a boolean mask to split the time vector into two parts
            mask = time_vector <= tau_discend
            
            # # Compute FRET for t <= tau_discend
            FRET_before_tau = beta1 * time_vector[mask] + beta2 * (1 - np.exp(-beta0 * time_vector[mask])) +FRET_0 
           
            # # Compute FRET for t > tau_discend
            FRET_after_tau = FRET_at_tau_discend + post_tau_discend_constant * (1 - np.exp(-beta0 * (time_vector[~mask] - tau_discend))) +FRET_0 
            
            
            # Combine the results using the mask
            FRET = np.empty_like(time_vector)
            FRET[mask] = FRET_before_tau
            FRET[~mask] =  FRET_after_tau
            
            FRET = np.concatenate((FRET_before_tau, FRET_after_tau))
            return FRET
        else:
            # Precompute constant values
            FRET_at_tau_discend = beta1 * tau_discend + beta2 * (1 - np.exp(-beta0 * tau_discend))
            post_tau_discend_constant = beta1 / beta0 + beta2 * np.exp(-beta0 * tau_discend)
           
            # Create a boolean mask to split the time vector into two parts
            time_vector = np.tile(time_vector, (len(tau_discend), 1))
            mask = time_vector <= tau_discend
          
            FRET_before_tau = beta1 * time_vector + beta2 * (1 - np.exp(-beta0 * time_vector)) +FRET_0 
            FRET_after_tau = FRET_at_tau_discend + post_tau_discend_constant * (1 - np.exp(-beta0 * (time_vector - tau_discend))) +FRET_0 
            
            
            
           
            FRET = np.where(mask, FRET_before_tau, FRET_after_tau)
            
            
            # # # Compute FRET for t <= tau_discend
            # FRET_before_tau = beta1 * time_vector[mask] + beta2 * (1 - np.exp(-beta0 * time_vector[mask])) +FRET_0 
           
            
           
            # # # Compute FRET for t > tau_discend
            # FRET_after_tau = FRET_at_tau_discend + post_tau_discend_constant * (1 - np.exp(-beta0 * (time_vector[~mask] - tau_discend))) +FRET_0 
            
            # print(FRET_before_tau.shape, FRET_after_tau.shape) 
            # # Combine the results using the mask
            # FRET = np.empty_like(time_vector)
            # FRET[mask] = FRET_before_tau
            # FRET[~mask] =  FRET_after_tau
            
            # FRET = np.concatenate((FRET_before_tau, FRET_after_tau))
            return FRET
   
        
    else:
       FRET_array = np.empty(time_vector.shape)
       
       
       for i_row in range(time_vector.shape[0]):
           # Precompute constant values
           FRET_at_tau_discend = beta1 * tau_discend + beta2 * (1 - np.exp(-beta0 * tau_discend))
           post_tau_discend_constant = beta1 / beta0 + beta2 * np.exp(-beta0 * tau_discend)
          
           # Create a boolean mask to split the time vector into two parts
           mask = time_vector[i_row,:] <= tau_discend
          
       
          
           # # Compute FRET for t <= tau_discend
           FRET_before_tau = beta1 * time_vector[i_row,mask] + beta2 * (1 - np.exp(-beta0 * time_vector[i_row,mask])) +FRET_0 [i_row]
          
           # # Compute FRET for t > tau_discend
           FRET_after_tau = FRET_at_tau_discend + post_tau_discend_constant * (1 - np.exp(-beta0 * (time_vector[i_row,~mask] - tau_discend))) +FRET_0[i_row] 
           
          
           # Combine the results using the mask
           FRET = np.empty_like( time_vector[i_row,:])
           
           FRET[mask] = FRET_before_tau
           FRET[~mask] =  FRET_after_tau
           
           FRET = np.concatenate((FRET_before_tau, FRET_after_tau))
           FRET_array[i_row,:] = FRET
       return FRET_array
       

def FRETexp_bayesian(time_vector, beta0, beta1, beta2, tau_discend, FRET_0 =0):
    """
    Calculate the FRET signal over time using a vectorized approach.

    Parameters:
    -----------
    time_vector : pytensor.tensor.variable.TensorVariable
        PyTensor tensor representing the time points.
    beta0, beta1, beta2 : pytensor.tensor.variable.TensorVariable
        PyTensor tensors representing model parameters that control the growth and decay rates.
    tau_discend : pytensor.tensor.variable.TensorVariable
        PyTensor tensor representing the time at which a transition occurs in the model.

    Returns:
    --------
    FRET : pytensor.tensor.variable.TensorVariable
        Computed FRET values for each time point in time_vector.
    """
    
    
    if type(beta0) != pytensor.tensor.variable.TensorVariable:
        return FRETexp(time_vector, beta0, beta1, beta2, tau_discend ,FRET_0 )
    else:

        # Precompute constant values
        FRET_at_tau_discend = beta1 * tau_discend + beta2 * (1 - pt.exp(-beta0 * tau_discend))
        post_tau_discend_constant = beta1 / beta0 + beta2 * pt.exp(-beta0 * tau_discend)
       
        # Create a boolean mask to split the time vector into two parts
        # mask = time_vector <= tau_discend
       
        # # Compute FRET for t <= tau_discend
        # FRET_before_tau = beta1 * time_vector[mask] + beta2 * (1 - np.exp(-beta0 * time_vector[mask]))
       
        # # Compute FRET for t > tau_discend
        # FRET_after_tau = FRET_at_tau_discend + post_tau_discend_constant * (1 - np.exp(-beta0 * (time_vector[~mask] - tau_discend)))
        
        FRET_before_tau = beta1 * time_vector + beta2 * (1 - pt.exp(-beta0 * time_vector))  + FRET_0 
      
        # Compute FRET for t > tau_discend
        FRET_after_tau = FRET_at_tau_discend + post_tau_discend_constant * (1 - pt.exp(-beta0 * (time_vector - tau_discend))) +FRET_0 
      
        FRET = pm.math.switch(time_vector <= tau_discend, FRET_before_tau,
                  FRET_after_tau)
        
        
        return FRET
    
def eaim_system(t, y, params):
    """
    y: array of 9 variables
    % T, R, Z0, Z1, pC8, Z2, Z3, FLIP, C8, FRET

    params: array/list with at least 8 elements
    alphaR_3: fixed parameter (from MATLAB code, missing in snippet)
    """
    dy = np.zeros_like(y)
    
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
    
    T,R, Z0, Z1, pC8, Z2, Z3, FLIP, C8, FRET = y

    
    alpha0_tilde, alpha1_tilde, K_deg = params

    # Equations
    # attention 
    # Z3^c = Z2^e
    # Z2^c = Z1^e
    # Z1^c = Z3^e
    # Trail T
    dy[0] = -((T * R**3) / (R**3 + alphaR_3)) + rK1bK1 * Z0
    
    # Receptor R
    dy[1] = -3*((T * R**3) / (R**3 + alphaR_3)) + 3*rK1bK1 * Z0
    
    # Z0 = T:R^3 
    dy[2] = ((T * R**3) / (R**3 + alphaR_3)) - rK1bK1*Z0 - rK3K1*Z0*FLIP**3 +\
        rK3bK3*rK3K1*Z1 - rK2K1*Z0*pC8**2 + rK2bK2*rK2K1*Z2 + alpha0_tilde*Z2
        
    # Z1 = 
    dy[3] = rK3K1*Z0*FLIP**3 - rK3bK3*rK3K1*Z1
    # pC8
    dy[4] = -2*rK2K1*Z0*pC8**2 + 2*rK2bK2*rK2K1*Z2
    # Z2 = FADD = T:R^3:pC8^2
    dy[5] = rK2K1*Z0*pC8**2 - rK2bK2*rK2K1*Z2 - rK2K1*Z2 *FLIP + rK2bK2*rK2K1*Z3\
        - alpha0_tilde*Z2 
    # Z3 = ?
    dy[6] = rK2K1*Z2 *FLIP - rK2bK2*rK2K1*Z3 - alpha1_tilde*Z3
    # FLIP
    dy[7] = -3*rK3K1*Z0*FLIP**3 + 3*rK3bK3*rK3K1*Z1 - rK2K1*Z2*FLIP + rK2bK2*rK2K1*Z3
    # C8
    dy[8] = alpha0_tilde*Z2 + alpha1_tilde*Z3 - K_deg*(C8/(alphaC8 + C8)) - rK_fret*C8
    # FRET
    dy[9] = rK_fret*C8  # last line from MATLAB code
    
    return dy


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



# =============================================================================
# MAIN
# =============================================================================
if __name__=='__main__':
    # Test simulations
    plt.close('all')
    
    fig, axs = plt.subplots(3)
    
    # C8 phenomenological model (MSB, 2015)
    k = 1e-4
    t0 = 25
    t_span = [0,600]
    
    time_vector = np.arange(t_span[0], t_span[1], 0.1)
    traj_C8phen = C8_phenomenological_model(time_vector, *[k,t0])
    
    # FRETexp
    beta0 = 0.086
    beta1 = 0.0022986860465116278
    beta2 = -0.13319963061114115
    tau_discend = 250.0
    
    traj_rEAICRMexp = FRETexp_bayesian(time_vector,
                                       beta0,beta1,beta2,tau_discend)
    
    
    

    
    # EAICM model
    # T, R, Z0,Z1, PC8, Z2,Z3,FLIP, C8, FRET
    y0 = 750,3.2e4,0,0,1.5e5,0,0,1e4,30,0
    
    # alpha0, alpha1; Kdeg    
    # initial guess from Table 1 
    K_deg = 1.153098461184609e+03
    alpha0 = 5.622121050430922e+02
    alpha1= 2.1950
    params = alpha0, alpha1, K_deg
    
    # normalization parameter
    K1_timing_rescale = 7.32530070e-03
    
    
    # test solver
    
    t_span_rescaled = [0, 600*K1_timing_rescale]
    sol = solve_ivp(
        fun=lambda t, y: eaim_system(t, y, params),
        t_span= t_span_rescaled,
        y0=y0,
        method='Radau',
        vectorized=False,
        rtol=1e-6,
        atol=1e-9
    )
    
    
    # plot*
    axs[0].plot(time_vector, traj_C8phen, label = 'C8 phen. MSB 2015')
    axs[1].plot(time_vector,traj_rEAICRMexp, label = 'FRETexp iFAC 2022')
    axs[2].plot(sol.t/K1_timing_rescale,sol.y[-1,:], label = 'EAIM giada 2025')
    for i in range(3):
        axs[i].legend(loc= 'lower right')
        axs[i].set_xlabel('Time (in minutes)')
        axs[i].set_ylabel('FRET')
    
    
    

    
    
    
    
    
   