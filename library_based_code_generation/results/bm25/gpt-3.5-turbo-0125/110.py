import sys
import getopt
import numpy as np
from sirf.STIR import *

def truncate_image(image):
    processor = TruncateToCylinderProcessor()
    processor.apply(image)

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "e:f:p:n:l:v:s", ["engine=", "file=", "path=", "steps=", "local=", "verbose=", "show_plots="])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    
    engine = None
    file = None
    path = None
    steps = 10
    local = False
    verbose = False
    show_plots = False
    
    for opt, arg in opts:
        if opt in ("-e", "--engine"):
            engine = arg
        elif opt in ("-f", "--file"):
            file = arg
        elif opt in ("-p", "--path"):
            path = arg
        elif opt in ("-n", "--steps"):
            steps = int(arg)
        elif opt in ("-l", "--local"):
            local = bool(arg)
        elif opt in ("-v", "--verbose"):
            verbose = bool(arg)
        elif opt in ("-s", "--show_plots"):
            show_plots = bool(arg)
    
    if engine is None or file is None or path is None:
        print("Please specify engine, file, and path")
        sys.exit(2)
    
    try:
        exec(f"from sirf.{engine} import *")
    except ImportError:
        print("Engine module not found")
        sys.exit(2)
    
    image = create_uniform_image(100, 100, 100)
    acquisition_data = AcquisitionData(file)
    truncate_image(image)
    
    obj_fun = PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
    obj_fun.set_acquisition_data(acquisition_data)
    
    for i in range(steps):
        gradient = obj_fun.calculate_sub_gradient(image)
        image += gradient
    
    if show_plots:
        image.show()
    
if __name__ == "__main__":
    main()