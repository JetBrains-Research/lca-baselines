 Here is a simplified Python script that follows your instructions. Note that this script assumes you have the SIRF library installed and the necessary command-line parsing library (argparse).

```python
import argparse
import sys
from sirflab.engines import CartesianEngine, RadialEngine, GoldenAngleEngine, GRPEEngine
from sirflab.acquisition import Acquisition
from sirflab.reconstruction import ConjugateGradient
from sirflab.operators import SymmetricalOperator

def create_symmetrical_operator(data):
    # Define your symmetrical operator here
    pass

def conjugate_gradient(data, operator, acquisition_model, sensitivity_maps):
    # Implement the Conjugate Gradient method here
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True, help='Path to the raw data file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data files')
    parser.add_argument('--output_file', type=str, help='Output file for simulated data')
    parser.add_argument('--engine', type=str, choices=['cartesian', 'radial', 'goldenangle', 'grpe'], default='cartesian', help='Reconstruction engine')
    parser.add_argument('--non_cartesian', action='store_true', help='Run the reconstruction if non-cartesian code was compiled')
    parser.add_argument('--trajectory', type=str, choices=['cartesian', 'radial', 'goldenangle', 'grpe'], default='cartesian', help='Trajectory type')
    parser.add_argument('--plot', action='store_true', help='Show plots')

    args = parser.parse_args()

    engine_module = None
    if args.engine == 'cartesian':
        engine_module = CartesianEngine
    elif args.engine == 'radial':
        engine_module = RadialEngine
    elif args.engine == 'goldenangle':
        engine_module = GoldenAngleEngine
    elif args.engine == 'grpe':
        engine_module = GRPEEngine

    if engine_module is None:
        print("Invalid engine option.")
        sys.exit(1)

    acquisition = Acquisition.from_hdf5(args.data_path)
    operator = SymmetricalOperator(create_symmetrical_operator(acquisition.data))
    sensitivity_maps = acquisition.get_coil_sensitivities()
    acquisition_model = acquisition.get_acquisition_model()

    if args.non_cartesian and engine_module != RadialEngine and engine_module != GRPEEngine:
        print("Non-cartesian engine not supported for this data.")
        sys.exit(1)

    reconstruction = ConjugateGradient(operator, acquisition_model, sensitivity_maps)
    reconstruction.run(acquisition.data, args.plot)

if __name__ == "__main__":
    main()
```

This script defines the main structure of the program, but the `create_symmetrical_operator`, `conjugate_gradient`, and the actual implementation of the Conjugate Gradient method are left as placeholders for you to fill in according to your specific needs.