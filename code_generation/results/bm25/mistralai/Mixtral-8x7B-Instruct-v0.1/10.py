 import numpy as np
from discretize import TensorMesh
from SimPEG import (
    maps,
    surveys,
    data,
    inversion,
    optimization,
    potentials,
    physics,
)
from SimPEG.electromagnetics.frequency_domain import FDEM
from matplotlib import pyplot as plt

# Set up cylindrically symmetric mesh
Nc = 21
mesh = TensorMesh([Nc], "C")

# Define geologic parameters and electrical conductivity
sigma = 0.1
rho = 1 / sigma

# Set up relative magnetic permeability
mu_r = np.ones(mesh.nC)

# Define mappings
sigmaMap = maps.ExpMap(mesh)
muMap = maps.IdentityMap(mesh)
mapped_rho = sigmaMap * rho
mapped_mu_r = muMap * mu_r

# Set up FDEM problem and survey
problem = FDEM.Problem3D_CC(mesh)
problem.pair(mapped_rho)
problem.pair(mapped_mu_r)
survey = surveys.BaseFDEMSurvey(
    [surveys.bzLoop(freq, loc, size, amplitude) for freq, loc, size, amplitude in [[1, [0, 0, 0], 1, 1]]]
)

# Set up inversion directives
inversionDirs = {
    "b": 1.0,
    "max_iters": 50,
    "threshold": 1e-3,
    "tolerance": 1e-3,
    "command": "mumps",
    "max_linear_complexity": 100,
    "show_progress_bars": True,
}

# Perform FDEM inversion
invProb = inversion.BaseInversion(problem, survey)
invProb.setUp(**inversionDirs)
dataMisfit = inversion.dataMisfit.l2_DataMisfit(survey=survey)
simulation = potentials.TDEMSimulation3D_CC(problem)
invProb.set_simulation(simulation)

# Set up inversion directives
inv = ComplicatedInversion(invProb)
inv.set_inversion_directives(**inversionDirs)

# Run inversion
inv.run()

# Plot conductivity model, permeability model, and data misfits
if plot_results:
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(221)
    mesh.plot_image(inv.model, ax=ax1, v_type="CPR", cmap="viridis")
    ax1.set_title("Conductivity Model")

    ax2 = fig.add_subplot(222)
    mesh.plot_image(inv.mu, ax=ax2, v_type="CPR", cmap="viridis")
    ax2.set_title("Permeability Model")

    ax3 = fig.add_subplot(212)
    invProb.plot_dataMisfit(ax=ax3)
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Data Misfit")

    plt.show()