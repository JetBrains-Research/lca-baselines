 import argparse
import os
from sirf.ScatterEstimator import ScatterEstimator
from sirf.SingleScatterSimulator import SingleScatterSimulator
from sirf.STIR import TestSTIRAcquisitionDataAlgebraFile
from sirf.Utility import set_collimator_file, set_detector_file, set_parameter_file, read_from_file, set_attenuation_correction_factors, label_and_name, name_and_parameters, set_working_folder_file_overwrite, set_attenuation_image, normalise_zero_and_one, get_attenuation_image, PoissonLogLikelihoodWithLinearModelForMeanAndProjData, set_attenuation_type, get_attenuation_type, estimate_randoms, set_output_prefix
import matplotlib.pyplot as plt

def main():
 parser = argparse.ArgumentParser(description='Perform scatter estimation in PET imaging using SIRF.')
 parser.add_argument('raw_data_file', help='Path to the raw data file.')
 parser.add_argument('randoms_data_file', help='Path to the randoms data file.')
 parser.add_argument('attenuation_correction_factors_file', help='Path to the attenuation correction factors file.')
 parser.add_argument('path_to_normalization_and_attenuation_files', help='Path to the normalization and attenuation files.')
 parser.add_argument('normalization_file', help='Name of the normalization file.')
 parser.add_argument('attenuation_image_file', help='Path to the attenuation image file.')
 parser.add_argument('output_prefix', help='Output prefix for scatter estimates.')
 parser.add_argument('--non-interactive', action='store_true', help='Enable non-interactive mode.')
 args = parser.parse_args()

 try:
 set_working_folder_file_overwrite(os.getcwd())
 set_collimator_file(os.path.join(args.path_to_normalization_and_attenuation_files, 'collimator.mat'))
 set_detector_file(os.path.join(args.path_to_normalization_and_attenuation_files, 'detector.mat'))
 set_parameter_file(os.path.join(args.path_to_normalization_and_attenuation_files, 'parameters.mat'))
 set_attenuation_correction_factors(args.attenuation_correction_factors_file)
 set_attenuation_image(args.attenuation_image_file)

 acquisition = TestSTIRAcquisitionDataAlgebraFile(args.raw_data_file)
 randoms = TestSTIRAcquisitionDataAlgebraFile(args.randoms_data_file)

 normalised_acquisition = normalise_zero_and_one(acquisition)
 normalised_randoms = normalise_zero_and_one(randoms)

 simulator = SingleScatterSimulator()
 simulator.set_attenuation_type(get_attenuation_type(acquisition))
 simulator.set_attenuation_image(get_attenuation_image())
 simulator.set_parameter_file(os.path.join(args.path_to_normalization_and_attenuation_files, 'scatter_estimation_parameters.mat'))
 simulator.set_output_prefix(args.output_prefix)

 scatter_estimator = ScatterEstimator()
 scatter_estimator.set_simulator(simulator)
 scatter_estimator.set_acquisition(normalised_acquisition)
 scatter_estimator.set_randoms(normalised_randoms)
 scatter_estimator.set_normalisation(read_from_file(os.path.join(args.path_to_normalization_and_attenuation_files, args.normalization_file)))

 scatter_estimator.process()

 if not args.non_interactive:
 from sirf.STIR import ProjData
 proj_data = ProjData(os.path.join(args.output_prefix, 'scatter_estimate.hdr'))
 plt.plot(proj_data[0, :, 0])
 plt.show()

 except Exception as e:
 print(f'Error: {e}')

if __name__ == '__main__':
 main()