# -*- coding: utf-8 -*-
"""
Downloaded on Fri Nov 28 14:51:19 2025

@author: carlos martinez von dossow
"""

############################################
# 1. DATA
############################################

import pandas as pd

data = {
    't': [0.0, 2.5, 5.0, 7.5],
    'mA': [0.08597285067873303, 0.07651245551601424, 0.08464428269546842, 0.08262289134772356],
    'mB': [0.06462984723854288, 0.07211538461538461, 0.06740225473734708, 0.06538461538461539],
    'mC': [0.0688622754491018, 0.06178707224334601, 0.05149884704073789, 0.04095354523227384],
    'mD': [0.052980132450331126, 0.04288939051918736, 0.05454545454545454, 0.0365296803652968]
}

# Crear el DataFrame
D1 = pd.DataFrame(data)

data = {
    't': [0, 2.5, 5, 7.5],
    'mA': [0.08616230197, 0.07973733583, 0.08214528174, 0.07823355858],
    'mB': [0.07270101107, 0.07869205893, 0.0624164066, 0.06023825579],
    'mC': [0.04668674699, 0.06516290727, 0.04875621891, 0.04738154613],
    'mD': [0.1012269939, 0.07090150786, 0.05513307985, 0.04234527687]
}

# Crear el DataFrame
D2 = pd.DataFrame(data)

# Datos proporcionados
data = {
    't': [0, 2.5, 5, 7.5],
    'mA': [0.1011892764, 0.09164149043, 0.1081081081, 0.09538584726],
    'mB': [0.0796460177, 0.08024865781, 0.08011869436, 0.07144461919],
    'mC': [0.07976878613, 0.06712962963, 0.07431340872, 0.0527654164],
    'mD': [0.05357142857, 0.04444444444, 0.05078125, 0.04416403785]
}

# Crear el DataFrame
D3 = pd.DataFrame(data)

# Datos proporcionados
data = {
    't': [0, 2.5, 5, 7.5],
    'mA': [0.09095617943, 0.1033470346, 0.1115222442, 0.1257838436],
    'mB': [0.07793398533, 0.06284252832, 0.07558500725, 0.07159442724],
    'mC': [0.07671957672, 0.05902004454, 0.05590416429, 0.06447831184],
    'mD': [0.0618556701, 0.05698005698, 0.03992740472, 0.04020100503]
}

# Crear el DataFrame
L1 = pd.DataFrame(data)


# Datos proporcionados
data = {
    't': [0, 2.5, 5, 7.5],
    'mA': [0.08632210555, 0.1107196779, 0.1290882779, 0.1232241229],
    'mB': [0.07286876548, 0.06447831184, 0.06605922551, 0.07323340471],
    'mC': [0.06316725979, 0.07282913165, 0.07477477477, 0.04355716878],
    'mD': [0.05128205128, 0.03244837758, 0.03822937626, 0.03726708075]
}

# Crear el DataFrame
L2 = pd.DataFrame(data)

# Datos proporcionados
data = {
    't': [0, 2.5, 5, 7.5],
    'mA': [0.09725380444, 0.1095005239, 0.1288438413, 0.1328824142],
    'mB': [0.08237105466, 0.08556366586, 0.09006571318, 0.07479224377],
    'mC': [0.07831325301, 0.06806930693, 0.05044843049, 0.05428954424],
    'mD': [0.05504587156, 0.0608974359, 0.06857142857, 0.03070175439]
}

# Crear el DataFrame
L3 = pd.DataFrame(data)



############################################
# 2. Bayesian Model
############################################

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pymc.ode import DifferentialEquation

def fit_model(C1, C2, C3):
    # Convertir t_eval explícitamente a array fijo
    t_raw = C1['t'].to_numpy()
    t_eval = np.array(t_raw[1:], dtype=np.float64)

    # Datos experimentales
    C1_mA, C2_mA, C3_mA = C1['mA'].to_numpy(), C2['mA'].to_numpy(), C3['mA'].to_numpy()
    C1_mB, C2_mB, C3_mB = C1['mB'].to_numpy(), C2['mB'].to_numpy(), C3['mB'].to_numpy()
    C1_mC, C2_mC, C3_mC = C1['mC'].to_numpy(), C2['mC'].to_numpy(), C3['mC'].to_numpy()
    C1_mD, C2_mD, C3_mD = C1['mD'].to_numpy(), C2['mD'].to_numpy(), C3['mD'].to_numpy()

    # ODE compatible con PyMC
    def sys_ing_sunode(y, t, theta):
        R = theta[0]
        l = theta[1]
        mA = y[0]
        mB = y[1]
        mC = y[2]
        mD = y[3]
        return [
            R - l * mA,
            R / 2 - l * mB,
            R / 5 - l * mC,
            R / 10 - l * mD
        ]

    # ODE model con sunode
    ode_model = DifferentialEquation(
        func=sys_ing_sunode,
        times=t_eval,
        n_states=4,
        n_theta=2,
        t0=0
    )

    with pm.Model() as model:
        # Parámetros globales
        R = pm.Uniform("R", lower=0, upper=0.5)
        l = pm.Uniform("l", lower=0, upper=0.5)

        # Condiciones iniciales por réplica
        def lognormal_init(name, val):
            return pm.LogNormal(name, mu=np.log(val), sigma=0.1)

        mA0_1, mA0_2, mA0_3 = lognormal_init("mA0_1", C1_mA[0]), lognormal_init("mA0_2", C2_mA[0]), lognormal_init("mA0_3", C3_mA[0])
        mB0_1, mB0_2, mB0_3 = lognormal_init("mB0_1", C1_mB[0]), lognormal_init("mB0_2", C2_mB[0]), lognormal_init("mB0_3", C3_mB[0])
        mC0_1, mC0_2, mC0_3 = lognormal_init("mC0_1", C1_mC[0]), lognormal_init("mC0_2", C2_mC[0]), lognormal_init("mC0_3", C3_mC[0])
        mD0_1, mD0_2, mD0_3 = lognormal_init("mD0_1", C1_mD[0]), lognormal_init("mD0_2", C2_mD[0]), lognormal_init("mD0_3", C3_mD[0])

        sigma = pm.Uniform("sigma", lower=0, upper=0.1)

        # Observaciones por réplica
        def observe_batch(mA0, mB0, mC0, mD0, C_mA, C_mB, C_mC, C_mD, prefix):
            y0 = [mA0, mB0, mC0, mD0]
            sol = ode_model(y0=y0, theta=[R, l])
            pm.StudentT(f"{prefix}_mA", mu=sol[:, 0], sigma=sigma, nu=3, observed=C_mA[1:])
            pm.StudentT(f"{prefix}_mB", mu=sol[:, 1], sigma=sigma, nu=3, observed=C_mB[1:])
            pm.StudentT(f"{prefix}_mC", mu=sol[:, 2], sigma=sigma, nu=3, observed=C_mC[1:])
            pm.StudentT(f"{prefix}_mD", mu=sol[:, 3], sigma=sigma, nu=3, observed=C_mD[1:])

        observe_batch(mA0_1, mB0_1, mC0_1, mD0_1, C1_mA, C1_mB, C1_mC, C1_mD, "obs_C1")
        observe_batch(mA0_2, mB0_2, mC0_2, mD0_2, C2_mA, C2_mB, C2_mC, C2_mD, "obs_C2")
        observe_batch(mA0_3, mB0_3, mC0_3, mD0_3, C3_mA, C3_mB, C3_mC, C3_mD, "obs_C3")

        # Muestreo
        trace = pm.sample(tune=500, draws=1000, cores=4, chains=4)

    return trace

###########################################################################################
# 3. Run MCMC  (skip to next code if you want to load PyMC outputs instead of running MCMC)
#############################################################################################

trace_D = fit_model(D1, D2, D3)
summary = az.summary(trace_D, hdi_prob=0.95)
print(summary)


# Extract posterior samples for R, l, and sigma
R_samples_D = 24*trace_D.posterior['R'].values.flatten()
l_samples_D = 24*trace_D.posterior['l'].values.flatten()
sigma_samples_D = trace_D.posterior['sigma'].values.flatten()



trace_L = fit_model(L1, L2, L3)
summary = az.summary(trace_L, hdi_prob=0.95)
print(summary)


# Extract posterior samples for R, l, and sigma
R_samples_L = 24*trace_L.posterior['R'].values.flatten()
l_samples_L = 24*trace_L.posterior['l'].values.flatten()
sigma_samples_L = trace_L.posterior['sigma'].values.flatten()

############################################
# 4.  Load Outputs of PyMC
############################################

trace_D = az.from_netcdf("trace_D.nc")
trace_L = az.from_netcdf("trace_L.nc")


summary = az.summary(trace_D, hdi_prob=0.95)
print(summary)


# Extract posterior samples for R, l, and sigma
R_samples_D = 24*trace_D.posterior['R'].values.flatten()
l_samples_D = 24*trace_D.posterior['l'].values.flatten()
sigma_samples_D = trace_D.posterior['sigma'].values.flatten()

summary = az.summary(trace_L, hdi_prob=0.95)
print(summary)


# Extract posterior samples for R, l, and sigma
R_samples_L = 24*trace_L.posterior['R'].values.flatten()
l_samples_L = 24*trace_L.posterior['l'].values.flatten()
sigma_samples_L = trace_L.posterior['sigma'].values.flatten()


############################################
# 5. Histograms of main parameters
############################################

import matplotlib.pyplot as plt

# Calculate means
mean_R_D = np.mean(R_samples_D)
mean_R_L = np.mean(R_samples_L)
mean_l_D = np.mean(l_samples_D)
mean_l_L = np.mean(l_samples_L)

# Create a figure and a set of subplots (2 rows x 1 column)
fig, ax = plt.subplots(2, 1, figsize=(6, 5))

# Plot histograms for R samples
ax[0].hist(R_samples_D, bins=30, alpha=0.5, label='D', color='blue')
ax[0].hist(R_samples_L, bins=30, alpha=0.5, label='L', color='orange')
ax[0].axvline(mean_R_D, color='blue', linestyle='--', linewidth=1.5)
ax[0].axvline(mean_R_L, color='orange', linestyle='--', linewidth=1.5)
ax[0].set_title('Histograms of R samples')
ax[0].set_xlabel('R value')
ax[0].set_ylabel('Frequency')
ax[0].legend()

# Plot histograms for l samples
ax[1].hist(l_samples_D, bins=30, alpha=0.5, label='D', color='blue')
ax[1].hist(l_samples_L, bins=30, alpha=0.5, label='L', color='orange')
ax[1].axvline(mean_l_D, color='blue', linestyle='--', linewidth=1.5)
ax[1].axvline(mean_l_L, color='orange', linestyle='--', linewidth=1.5)
ax[1].set_title('Histograms of l samples')
ax[1].set_xlabel('l value')
ax[1].set_ylabel('Frequency')
ax[1].legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


############################################
# 6. Functions to make plots of fitting
############################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pymc as pm
import arviz as az
from scipy.stats import t

def plot_concentration(trace, C1, C2, C3, c1, c2, c3):
    # Extract posterior samples for mA0, mB0, mC0, mD0
    mA0_1_samples = trace.posterior['mA0_1'].values.flatten()
    mA0_2_samples = trace.posterior['mA0_2'].values.flatten()
    mA0_3_samples = trace.posterior['mA0_3'].values.flatten()

    mB0_1_samples = trace.posterior['mB0_1'].values.flatten()
    mB0_2_samples = trace.posterior['mB0_2'].values.flatten()
    mB0_3_samples = trace.posterior['mB0_3'].values.flatten()

    mC0_1_samples = trace.posterior['mC0_1'].values.flatten()
    mC0_2_samples = trace.posterior['mC0_2'].values.flatten()
    mC0_3_samples = trace.posterior['mC0_3'].values.flatten()

    mD0_1_samples = trace.posterior['mD0_1'].values.flatten()
    mD0_2_samples = trace.posterior['mD0_2'].values.flatten()
    mD0_3_samples = trace.posterior['mD0_3'].values.flatten()

    # Extract posterior samples for R, l, and sigma
    R_samples = trace.posterior['R'].values.flatten()
    l_samples = trace.posterior['l'].values.flatten()
    sigma_samples = trace.posterior['sigma'].values.flatten()

    # Define the ODE system
    def sys_ing(t, y, R, l):
        dy = np.zeros(4)
        dy[0] = R - l*y[0]  # mA
        dy[1] = R/2 - l*y[1]  # mB
        dy[2] = R/5 - l*y[2]  # mC
        dy[3] = R/10 - l*y[3]  # mD
        return dy

    # Time points for the ODE solution
    t_eval = C1['t'].to_numpy()

    # Number of samples
    num_samples = len(R_samples)

    # Initialize arrays to store results
    mA_solutions_C1 = np.zeros((num_samples, len(t_eval)))
    mB_solutions_C1 = np.zeros((num_samples, len(t_eval)))
    mC_solutions_C1 = np.zeros((num_samples, len(t_eval)))
    mD_solutions_C1 = np.zeros((num_samples, len(t_eval)))

    mA_solutions_C2 = np.zeros((num_samples, len(t_eval)))
    mB_solutions_C2 = np.zeros((num_samples, len(t_eval)))
    mC_solutions_C2 = np.zeros((num_samples, len(t_eval)))
    mD_solutions_C2 = np.zeros((num_samples, len(t_eval)))

    mA_solutions_C3 = np.zeros((num_samples, len(t_eval)))
    mB_solutions_C3 = np.zeros((num_samples, len(t_eval)))
    mC_solutions_C3 = np.zeros((num_samples, len(t_eval)))
    mD_solutions_C3 = np.zeros((num_samples, len(t_eval)))

    def solve_and_store_solutions(C, mA0_samples, mB0_samples, mC0_samples, mD0_samples, mA_solutions, mB_solutions, mC_solutions, mD_solutions):
        for i in range(num_samples):
            R_i = R_samples[i]
            l_i = l_samples[i]
            mA0_i = mA0_samples[i]
            mB0_i = mB0_samples[i]
            mC0_i = mC0_samples[i]
            mD0_i = mD0_samples[i]

            solution = solve_ivp(
                sys_ing,
                (0, np.max(t_eval)),
                [mA0_i, mB0_i, mC0_i, mD0_i],
                t_eval=t_eval,
                args=(R_i, l_i)
            )

            loc = solution.y[0]
            mA_solutions[i, :] = t.rvs(df=3, loc=loc, scale=sigma_samples[i], size=len(loc))
            loc = solution.y[1]
            mB_solutions[i, :] = t.rvs(df=3, loc=loc, scale=sigma_samples[i], size=len(loc))
            loc = solution.y[2]
            mC_solutions[i, :] = t.rvs(df=3, loc=loc, scale=sigma_samples[i], size=len(loc))
            loc = solution.y[3]
            mD_solutions[i, :] = t.rvs(df=3, loc=loc, scale=sigma_samples[i], size=len(loc))

    # Solve the ODE for each sample and store solutions based on c1, c2, c3
    if c1 == 1:
        solve_and_store_solutions(C1, mA0_1_samples, mB0_1_samples, mC0_1_samples, mD0_1_samples, mA_solutions_C1, mB_solutions_C1, mC_solutions_C1, mD_solutions_C1)
        C1_mA = C1['mA'].to_numpy()
        C1_mB = C1['mB'].to_numpy()
        C1_mC = C1['mC'].to_numpy()
        C1_mD = C1['mD'].to_numpy()
    else:
        C1_mA = C1_mB = C1_mC = C1_mD = None

    if c2 == 1:
        solve_and_store_solutions(C2, mA0_2_samples, mB0_2_samples, mC0_2_samples, mD0_2_samples, mA_solutions_C2, mB_solutions_C2, mC_solutions_C2, mD_solutions_C2)
        C2_mA = C2['mA'].to_numpy()
        C2_mB = C2['mB'].to_numpy()
        C2_mC = C2['mC'].to_numpy()
        C2_mD = C2['mD'].to_numpy()
    else:
        C2_mA = C2_mB = C2_mC = C2_mD = None

    if c3 == 1:
        solve_and_store_solutions(C3, mA0_3_samples, mB0_3_samples, mC0_3_samples, mD0_3_samples, mA_solutions_C3, mB_solutions_C3, mC_solutions_C3, mD_solutions_C3)
        C3_mA = C3['mA'].to_numpy()
        C3_mB = C3['mB'].to_numpy()
        C3_mC = C3['mC'].to_numpy()
        C3_mD = C3['mD'].to_numpy()
    else:
        C3_mA = C3_mB = C3_mC = C3_mD = None

    # Calculate HDI and mean
    def calculate_hdi_and_mean(samples):
        mean = np.mean(samples, axis=0)
        hdi = az.hdi(samples, hdi_prob=0.95)
        return mean, hdi

    mA_mean_C1, mA_hdi_C1 = calculate_hdi_and_mean(mA_solutions_C1) if c1 == 1 else (None, None)
    mB_mean_C1, mB_hdi_C1 = calculate_hdi_and_mean(mB_solutions_C1) if c1 == 1 else (None, None)
    mC_mean_C1, mC_hdi_C1 = calculate_hdi_and_mean(mC_solutions_C1) if c1 == 1 else (None, None)
    mD_mean_C1, mD_hdi_C1 = calculate_hdi_and_mean(mD_solutions_C1) if c1 == 1 else (None, None)

    mA_mean_C2, mA_hdi_C2 = calculate_hdi_and_mean(mA_solutions_C2) if c2 == 1 else (None, None)
    mB_mean_C2, mB_hdi_C2 = calculate_hdi_and_mean(mB_solutions_C2) if c2 == 1 else (None, None)
    mC_mean_C2, mC_hdi_C2 = calculate_hdi_and_mean(mC_solutions_C2) if c2 == 1 else (None, None)
    mD_mean_C2, mD_hdi_C2 = calculate_hdi_and_mean(mD_solutions_C2) if c2 == 1 else (None, None)

    mA_mean_C3, mA_hdi_C3 = calculate_hdi_and_mean(mA_solutions_C3) if c3 == 1 else (None, None)
    mB_mean_C3, mB_hdi_C3 = calculate_hdi_and_mean(mB_solutions_C3) if c3 == 1 else (None, None)
    mC_mean_C3, mC_hdi_C3 = calculate_hdi_and_mean(mC_solutions_C3) if c3 == 1 else (None, None)
    mD_mean_C3, mD_hdi_C3 = calculate_hdi_and_mean(mD_solutions_C3) if c3 == 1 else (None, None)

    # Plot the results
    fig, axs = plt.subplots(2, 2, figsize=(10,7), sharex=True, sharey=True)

    def plot_with_hdi_and_data(ax, t_eval, mean_C1, hdi_C1, observed_data_C1, mean_C2, hdi_C2, observed_data_C2, mean_C3, hdi_C3, observed_data_C3, label):
        if mean_C1 is not None:
            ax.plot(t_eval, mean_C1, label='L1 Mean', color='blue')
            ax.fill_between(t_eval, hdi_C1[:, 0], hdi_C1[:, 1], color='blue', alpha=0.1)
            ax.plot(t_eval, observed_data_C1, 'o', color='blue', markersize=5)

        if mean_C2 is not None:
            ax.plot(t_eval, mean_C2, label='L2 Mean', color='red')
            ax.fill_between(t_eval, hdi_C2[:, 0], hdi_C2[:, 1], color='red', alpha=0.1)
            ax.plot(t_eval, observed_data_C2, 's', color='red', markersize=5)

        if mean_C3 is not None:
            ax.plot(t_eval, mean_C3, label='L3 Mean', color='green')
            ax.fill_between(t_eval, hdi_C3[:, 0], hdi_C3[:, 1], color='green', alpha=0.1)
            ax.plot(t_eval, observed_data_C3, 's', color='green', markersize=5)

        ax.set_xlabel('Time (t_eval)')
        ax.set_ylabel('Concentration')
        ax.set_title(label)
        ax.legend()
        ax.grid(True)

    plot_with_hdi_and_data(axs[0, 0], t_eval, mA_mean_C1, mA_hdi_C1, C1_mA, mA_mean_C2, mA_hdi_C2, C2_mA, mA_mean_C3, mA_hdi_C3, C3_mA, 'Concentration mA')
    plot_with_hdi_and_data(axs[0, 1], t_eval, mB_mean_C1, mB_hdi_C1, C1_mB, mB_mean_C2, mB_hdi_C2, C2_mB, mB_mean_C3, mB_hdi_C3, C3_mB, 'Concentration mB')
    plot_with_hdi_and_data(axs[1, 0], t_eval, mC_mean_C1, mC_hdi_C1, C1_mC, mC_mean_C2, mC_hdi_C2, C2_mC, mC_mean_C3, mC_hdi_C3, C3_mC, 'Concentration mC')
    plot_with_hdi_and_data(axs[1, 1], t_eval, mD_mean_C1, mD_hdi_C1, C1_mD, mD_mean_C2, mD_hdi_C2, C2_mD, mD_mean_C3, mD_hdi_C3, C3_mD, 'Concentration mD')

    # Adjust layout
    plt.tight_layout()
    plt.show()

############################################
# 7. Plots of fittings
############################################

plot_concentration(trace_D, D1, D2, D3, 0,1,0)

plot_concentration(trace_L, L1, L2, L3, 1,1,1)

