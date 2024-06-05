import sys
from sirf.STIR import *

def scatter_estimation(raw_data_file, randoms_data_file, attenuation_correction_factors_file, path_to_norm_atten_files, norm_file, atten_image_file, output_prefix, non_interactive_mode):
    try:
        scatter_est = ScatterEstimator()
        scatter_est.set_input(raw_data_file)
        scatter_est.set_randoms(randoms_data_file)
        scatter_est.set_attenuation_correction_factors(attenuation_correction_factors_file)
        scatter_est.set_normalisation(norm_file)
        scatter_est.set_attenuation_image(atten_image_file)
        scatter_est.set_output_prefix(output_prefix)
        
        if non_interactive_mode:
            scatter_est.set_verbosity(0)
        
        scatter_est.set_up()
        scatter_est.process()
        
        if not non_interactive_mode:
            scatter_est.get_output()
            scatter_est.plot()
        
    except error as err:
        print(f"Error: {err}")

if __name__ == '__main__':
    raw_data_file = sys.argv[1]
    randoms_data_file = sys.argv[2]
    attenuation_correction_factors_file = sys.argv[3]
    path_to_norm_atten_files = sys.argv[4]
    norm_file = sys.argv[5]
    atten_image_file = sys.argv[6]
    output_prefix = sys.argv[7]
    non_interactive_mode = bool(sys.argv[8])
    
    scatter_estimation(raw_data_file, randoms_data_file, attenuation_correction_factors_file, path_to_norm_atten_files, norm_file, atten_image_file, output_prefix, non_interactive_mode)