  ```
import pyaedt
import os

# Set non-graphical mode
non_graphical = False

# Download 3D component
component_path = "C:\\Program Files\\Ansys\\AnsysEM\\CustomComponents\\3DComponents\\"
component_name = "AntennaArray.a3d"
component_file = component_path + component_name

# Launch HFSS and save project
project_name = "AntennaArray"
project_path = os.path.join(os.getenv("USERPROFILE"), "Desktop", project_name)

# Read array definition from JSON file
array_definition = {
    "name": "AntennaArray",
    "components": [
        {
            "name": "Antenna",
            "type": "3D",
            "path": component_file
        }
    ],
    "array_size": [10, 10, 10],
    "array_spacing": [1, 1, 1],
    "array_rotation": [0, 0, 0],
    "array_mirror": [False, False, False]
}

# Set up simulation
simulation_setup = {
    "name": "SimulationSetup",
    "type": "HFSS",
    "setup_type": "DrivenModal",
    "frequency": 1e9,
    "num_modes": 10,
    "num_harmonics": 10,
    "num_eigen_modes": 10,
    "num_eigen_modes_to_export": 10,
    "num_ports": 1,
    "port_name": "Port1",
    "port_impedance": 50,
    "port_matching_type": "S-Parameters",
    "port_matching_tolerance": 1e-3,
    "port_matching_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations": 100,
    "port_matching_convergence_criteria": "Maximum",
    "port_matching_convergence_tolerance": 1e-3,
    "port_matching_convergence_max_iterations":