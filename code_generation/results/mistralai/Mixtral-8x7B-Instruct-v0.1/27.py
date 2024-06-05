 ```python
import dream
import matplotlib.pyplot as plt

# Initialize DREAM simulation
eqsys = dream.EquilibriumSystem()
eqsys.addions(["D", "T", "He", "He2", "He3", "W", "W+", "Fe", "Fe2+", "Fe12+", "Fe13+", "Fe14+", "Fe15+", "Fe16+", "Fe17+", "Fe18+", "Fe19+", "Fe20+", "Fe21+", "Fe22+"])
eqsys.set_radialgrid(nr=100, rmin=0.0, rmax=2.0)
eqsys.set_timesteps(nsteps=1000, dt=1e-7)
eqsys.set_boundaryconditions(divergent=False)
eqsys.set_initialprofiles(ne=1e19, ti=100, te=100, ni=eqsys.n_ions, vi=0.0)
eqsys.set_electricfield(phi=0.0)
eqsys.set_runawaygrid(nt_min=1, nt_max=1e25)
eqsys.set_hottailgrid(ne_hot=1e19)
eqsys.set_collisionfrequency(ioncoll=True, elastic=True, ionization=True, excitation=True, threebody=True, radiative=True)

# Initialize DREAM object
plas = dream.Plasma(eqsys)

# Initialize profiles
plas.initialize()

# Ionization stage
plas.ionizationstage()
plas.save("ionization")

# Equilibration stage
plas.equilibrationstage()
plas.save("equilibration")

# Radiation stage
plas.radiationstage()
plas.save("radiation")

# Plot ohmic heating and radiative losses as a function of temperature
T = plas.eqsys.get_profiles()["te"]
OhmicHeating = plas.eqsys.get_heatingrates()["ohmic"]
RadiativeLosses = plas.eqsys.get_coolingrates()["radiative"]
plt.plot(T, OhmicHeating, label="Ohmic Heating")
plt.plot(T, RadiativeLosses, label="Radiative Losses")
plt.xlabel("Temperature (eV)")
plt.ylabel("Heating/Cooling Rate (W/m^3)")
plt.legend()
plt.show()
```
Please note that the DREAM library must be installed and properly configured for this code to run. The above code assumes that the DREAM library is installed in the system and the required ions are available in the ion database.