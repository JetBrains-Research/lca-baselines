import sys
import numpy as np
from sirf.STIR import *

def OSMAPOSL_reconstruction(image, obj_fun, prior, filt, num_subsets, num_subiters):
    recon = OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(num_subsets)
    recon.set_num_subiterations(num_subiters)
    recon.set_anatomical_prior(prior)
    recon.set_input(image)
    recon.set_output_filename_prefix('recon')
    recon.set_up(filt)
    recon.reconstruct()

def main(raw_data_file, data_path, num_subsets, num_subiters, recon_engine, disable_plots):
    try:
        data_file = AcquisitionData(raw_data_file)
        acq_model = AcquisitionModelUsingRayTracingMatrix()
        acq_model.set_up(data_file, recon_engine)
        
        filter = TruncateToCylinderProcessor()
        filter.apply(acq_model.get_backprojection())
        
        init_image = acq_model.get_backprojection().clone()
        
        prior = QuadraticPrior()
        obj_fun = PoissonLogLikelihoodWithLinearModelForMean()
        obj_fun.set_acquisition_model(acq_model)
        
        OSMAPOSL_reconstruction(init_image, obj_fun, prior, filter, num_subsets, num_subiters)
        
        if not disable_plots:
            init_image.show()
    
    except error as err:
        print(err.what())

if __name__ == '__main__':
    raw_data_file = sys.argv[1]
    data_path = sys.argv[2]
    num_subsets = int(sys.argv[3])
    num_subiters = int(sys.argv[4])
    recon_engine = sys.argv[5]
    disable_plots = False if len(sys.argv) < 7 else sys.argv[6]
    
    main(raw_data_file, data_path, num_subsets, num_subiters, recon_engine, disable_plots)