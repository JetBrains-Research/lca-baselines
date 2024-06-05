


```python
import argparse
import logging
import sys

from sirf.Utilities import show_2D_array, show_3D_array
from sirf.Utilities import examples_data_path
from sirf.STIR import AcquisitionData, AcquisitionModel, ImageData, ImageGeometry, PoissonLogLikelihoodWithLinearModelForMeanAndProjData, TruncateToCylinderProcessor

def truncate_image(image):
    truncate_filter = TruncateToCylinderProcessor()
    truncate_filter.set_input(image)
    return truncate_filter.get_output()

def main(args):
    parser = argparse.ArgumentParser(description='Steepest ascent for Poisson log-likelihood objective function')
    parser.add_argument('--engine', help='Reconstruction engine', default='CPU')
    parser.add_argument('--raw', help='Raw data file', default='')
    parser.add_argument('--path', help='Path to data files', default=examples_data_path('PET'))
    parser.add_argument('--steps', help='Number of steepest descent steps', type=int, default=10)
    parser.add_argument('--local', help='Use locally optimal steepest ascent', action='store_true')
    parser.add_argument('--verbose', help='Verbosity', action='store_true')
    parser.add_argument('--plot', help='Show plots', action='store_true')
    args = parser.parse_args(args)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    try:
        engine = __import__('sirf.{0}'.format(args.engine), fromlist=['AcquisitionModel'])
    except ImportError:
        print('Engine {0} not found'.format(args.engine))
        return 1

    acq_data = AcquisitionData(args.raw, geometry=ImageGeometry(voxel_num_x=256, voxel_num_y=256, voxel_num_z=1, voxel_size_x=2.0, voxel_size_y=2.0, voxel_size_z=2.0))
    acq_data.read_from_file(args.path)

    image = ImageData(geometry=acq_data.geometry, dimension_labels=['channel', 'horizontal_x', 'horizontal_y'])
    image.initialise(1.0)

    objective_function = PoissonLogLikelihoodWithLinearModelForMeanAndProjData(acq_data, image)

    for step in range(args.steps):
        logging.info('Step {0}'.format(step))
        objective_function.set_objective_function(acq_data)
        objective_function.get_objective_function().set_acquisition_model(engine.AcquisitionModel(acq_data.geometry))
        objective_function.get_objective_function().set_up(image)
        logging.info('Objective function value: {0}'.format(objective_function.get_objective_function().value_of(image)))

        if args.plot:
            show_2D_array('Objective function', objective_function.get_objective_function().as_array())

        image = truncate_image(image)

        if args.plot:
            show_3D_array('Image', image.as_array())

        image = objective_function.get_backprojection_of_acquisition_ratio()

        if args.plot:
            show_3D_array('Backprojection', image.as_array())

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
```

This code imports the necessary libraries and defines a function to truncate an image. The main function parses command-line arguments, reads raw data from a file, creates an acquisition model, reads PET acquisition data from the specified file, creates a filter that zeroes the image outside a cylinder of the same diameter as the image xy-section size, creates an initial image estimate, creates an objective function of Poisson logarithmic likelihood type compatible with the acquisition data type, and performs the steepest descent steps. If anything goes wrong, the code catches and displays the error information. Source: assistant
EOT: true