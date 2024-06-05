```python
# Set up cylindrically symmetric mesh
mesh = Mesh.CylMesh([[(10, 32)], 32, [(10, 32)]], '00C')

# Define geologic parameters and electrical conductivity
sigma = np.ones(mesh.nC) * 1e-2
chi = np.ones(mesh.nC) * 0.01

# Set up relative magnetic permeability
mu = np.ones(mesh.nC)

# Define mappings
mapping = Maps.ExpMap(mesh)

# Set up FDEM problem and survey
prob = FDEM.Problem3D_e(mesh, sigma=sigma, mu=mu)
survey = FDEM.Survey(prob)

# Perform FDEM inversion
dmis = DataMisfit.l2_DataMisfit(survey)
reg = Regularization.Simple(mesh)
opt = Optimization.InexactGaussNewton(maxIter=10)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
inv = Inversion.BaseInversion(invProb)
prob.pair(survey)
m0 = np.ones(mesh.nC) * 1e-4
mopt = inv.run(m0)

# Set up inversion directives
directiveList = []
directiveList.append(Directives.BetaSchedule(coolingFactor=2, coolingRate=1))
directiveList.append(Directives.TargetMisfit())
directiveList.append(Directives.BetaEstimate_ByEig())

# Run the inversion
inv.directiveList = directiveList
mrec = inv.run(m0)

# Plot the conductivity model, permeability model, and data misfits if flag is set to true
if plot_flag:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    mesh.plotImage(np.log10(sigma), ax=ax[0])
    ax[0].set_title('Conductivity Model')
    mesh.plotImage(chi, ax=ax[1])
    ax[1].set_title('Permeability Model')
    survey.dobs = survey.dpred(mopt)
    Utils.plot2Ddata(survey.srcList[0].rxList[0].locs, survey.dobs, ax=ax[2])
    ax[2].set_title('Data Misfits')
    plt.show()
```