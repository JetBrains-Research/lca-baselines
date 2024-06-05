

```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import discretize
from SimPEG import (
    maps,
    utils,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
    data,
)
from SimPEG.electromagnetics import frequency_domain as fdem

# Define the mesh
cs, ncx, ncy, ncz, npad = 5.0, 25, 25, 25, 5
hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
mesh = discretize.TensorMesh([hx, hy, hz], "CCC")

# Define geologic parameters and electrical conductivity
sigma = np.ones(mesh.nC) * 1e-8  # electrical conductivity
sigma[mesh.gridCC[:, 2] < 0.0] = 1e-2  # add a conductive layer
sigma[mesh.gridCC[:, 2] < -100.0] = 1e-1  # add a deeper conductive layer
sigma[mesh.gridCC[:, 2] < -200.0] = 1e-4  # add a deeper conductive layer

mu = mu_0 * np.ones(mesh.nC)  # magnetic permeability
mu_r = 100.0 * np.ones(mesh.nC)  # relative magnetic permeability
mu[mesh.gridCC[:, 2] < 0.0] = mu_0 * mu_r[mesh.gridCC[:, 2] < 0.0]  # add a permeable layer

# Define mappings
actmap = maps.InjectActiveCells(mesh, ~air_inds, np.log(1e-8))  # active cells mapping
expmap = maps.ExpMap(mesh)  # mapping for electrical conductivity

# Set up the FDEM problem and survey
survey = fdem.Survey(source_list)  # define the survey
prob = fdem.Simulation3DMagneticField(
    mesh, survey=survey, sigmaMap=expmap * actmap, muMap=maps.IdentityMap(mesh)
)  # define the problem

# Perform the FDEM inversion
dmis = data_misfit.L2DataMisfit(data=data, simulation=prob)  # data misfit
reg = regularization.Sparse(mesh, indActive=~air_inds, mapping=maps.IdentityMap(nP=int(actv.sum())))  # regularization
opt = optimization.InexactGaussNewton(maxIter=10)  # optimization algorithm
invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)  # inverse problem

# Set up inversion directives
betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e0)  # estimate the trade-off parameter
target = directives.TargetMisfit()  # stop when we hit the target misfit
inv = inversion.BaseInversion(invProb, directiveList=[betaest, target])  # inversion

# Run the inversion
m0 = np.log(1e-8) * np.ones(actmap.nP)  # starting model
mrec = inv.run(m0)  # recovered model

# Plot the conductivity model, the permeability model, and the data misfits
if plot_flag:
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    plt.subplot(131)
    mesh.plot_slice(
        expmap * mrec,
        normal="Y",
        ax=ax[0],
        ind=int(mesh.hy.size / 2),
        grid=True,
        clim=(1e-3, 1e-1),
        pcolor_opts={"cmap": "viridis"},
    )
    plt.title("Conductivity Model")

    plt.subplot(132)
    mesh.plot_slice(
        mu_r,
        normal="Y",
        ax=ax[1],
        ind=int(mesh.hy.size / 2),
        grid=True,
        clim=(1, 100),
        pcolor_opts={"cmap": "viridis"},
    )
    plt.title("Relative Permeability Model")

    plt.subplot(133)
    plt.plot(invProb.dpred, "b", label="predicted")
    plt.plot(dobs, "r", label="observed")
    plt.legend(loc=1)
    plt.title("Data Misfit")

    plt.tight_layout()
    plt.show()
```

Output:

```

```

This code sets up a cylindrically symmetric mesh, defines geologic parameters and electrical conductivity, and sets up the relative magnetic permeability. It then sets up the FDEM problem and survey, performs the FDEM inversion, sets up inversion directives, and runs the inversion. If a flag is set to true, the code plots the conductivity model, the permeability model, and the data misfits.