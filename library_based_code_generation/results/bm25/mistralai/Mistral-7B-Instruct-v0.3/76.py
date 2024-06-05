 Here is a simplified example of how you might create a custom Li-ion model using the pybamm library. Note that this is a basic structure and you may need to adjust it according to your specific requirements.

```python
import pybamm

class MyLiIonModel(pybamm.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # External circuit
        self.external_circuit = pybamm.ExternalCircuit()

        # Current collector
        self.current_collector = pybamm.CurrentCollector()

        # Thermal
        self.thermal = pybamm.Thermal()

        # Porosity
        self.porosity = pybamm.Porosity()

        # Electrolyte diffusion
        self.electrolyte_diffusion = pybamm.ElectrolyteDiffusion()

        # Electrolyte conductivity
        self.electrolyte_conductivity = pybamm.ElectrolyteConductivity()

        # SEI
        self.sei = pybamm.SEI()

        # SEI on cracks
        self.sei_on_cracks = pybamm.SEIOnCracks()

        # Lithium plating
        self.lithium_plating = pybamm.LithiumPlating()

        # Negative electrode domains
        self.negative_electrode = pybamm.Domain(0, 1)
        self.negative_electrode_active_material = pybamm.ActiveMaterial()
        self.negative_electrode_electrode_potential = pybamm.ElectrodePotential()
        self.negative_electrode_particle = pybamm.Particle()
        self.negative_electrode_total_particle_concentration = pybamm.TotalParticleConcentration()
        self.negative_electrode_open_circuit_potential = pybamm.OpenCircuitPotential()
        self.negative_electrode_interface = pybamm.Interface()
        self.negative_electrode_interface_utilisation = pybamm.InterfaceUtilisation()
        self.negative_electrode_interface_current = pybamm.InterfaceCurrent()
        self.negative_electrode_surface_potential_difference = pybamm.SurfacePotentialDifference()
        self.negative_electrode_particle_mechanics = pybamm.ParticleMechanics()

        # Positive electrode domains
        self.positive_electrode = pybamm.Domain(0, 1)
        self.positive_electrode_active_material = pybamm.ActiveMaterial()
        self.positive_electrode_electrode_potential = pybamm.ElectrodePotential()
        self.positive_electrode_particle = pybamm.Particle()
        self.positive_electrode_total_particle_concentration = pybamm.TotalParticleConcentration()
        self.positive_electrode_open_circuit_potential = pybamm.OpenCircuitPotential()
        self.positive_electrode_interface = pybamm.Interface()
        self.positive_electrode_interface_utilisation = pybamm.InterfaceUtilisation()
        self.positive_electrode_interface_current = pybamm.InterfaceCurrent()
        self.positive_electrode_surface_potential_difference = pybamm.SurfacePotentialDifference()
        self.positive_electrode_particle_mechanics = pybamm.ParticleMechanics()

    def build(self):
        self._check_and_prepare_model_inplace()
        self.build_model()

    def process_model(self):
        self.process_parameters_and_discretise()
        self.check_and_set_domains()
        self.process_and_check()

    def solve(self):
        self.process_model()
        self.set_mesh()
        self.discretise_model()
        self.solve_model()

    def plot_solution(self):
        variables_and_events = self.variables_and_events()
        for var in variables_and_events:
            var.plot()

my_li_ion_model = MyLiIonModel()
my_li_ion_model.set_up_and_parameterise_model_for_experiment()
my_li_ion_model.build()
my_li_ion_model.process_model()
my_li_ion_model.solve()
my_li_ion_model.plot_solution()
```

This code creates a custom Li-ion model named `MyLiIonModel` with submodels for the components you specified. It then builds, processes the model, solves it, and plots the solution.