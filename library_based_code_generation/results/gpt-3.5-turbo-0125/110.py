import sys
import getopt
import numpy as np
from sirf.engine import specified_engine_module

def truncate_image(image):
    # Function to truncate the image
    pass

def main():
    # Process command-line options
    reconstruction_engine = ''
    raw_data_file = ''
    path_to_data_files = ''
    num_steps = 0
    use_locally_optimal_steepest_ascent = False
    verbosity = 0
    show_plots = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], "e:f:p:n:l:v:s", ["engine=", "file=", "path=", "steps=", "locally_optimal=", "verbosity=", "show_plots="])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-e", "--engine"):
            reconstruction_engine = arg
        elif opt in ("-f", "--file"):
            raw_data_file = arg
        elif opt in ("-p", "--path"):
            path_to_data_files = arg
        elif opt in ("-n", "--steps"):
            num_steps = int(arg)
        elif opt in ("-l", "--locally_optimal"):
            use_locally_optimal_steepest_ascent = True if arg.lower() == 'true' else False
        elif opt in ("-v", "--verbosity"):
            verbosity = int(arg)
        elif opt in ("-s", "--show_plots"):
            show_plots = True if arg.lower() == 'true' else False

    # Import specified engine module
    engine = specified_engine_module(reconstruction_engine)

    # Create acquisition model
    acquisition_model = engine.create_acquisition_model()

    # Read PET acquisition data
    data = engine.read_data(raw_data_file)

    # Create filter
    filter = engine.create_filter()

    # Create initial image estimate
    initial_image = engine.create_initial_image()

    # Create objective function
    objective_function = engine.create_objective_function(data)

    # Perform steepest descent steps
    for i in range(num_steps):
        # Perform steepest ascent step
        pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e))