  import sys
from sirtc.scatter_estimator import ScatterEstimator
from sirtc.single_scatter_simulator import SingleScatterSimulator
from sirtc.util import set_collimator_file, set_detector_file, set_parameter_file, read_from_file, set_attenuation_correction_factors, label_and_name, name_and_parameters, TestSTIRAcquisitionDataAlgebraFile, set_working_folder_file_overwrite, set_attenuation_image, process, normalise_zero_and_one, get_attenuation_image, PoissonLogLikelihoodWithLinearModelForMeanAndProjData, set_attenuation_type, get_attenuation_type, estimate_randoms, set_output_prefix

def main():
    # Parse command-line arguments
    args = sys.argv[1:]
    if len(args) != 10:
        print("Usage: python scatter_estimation.py <raw_data_file> <randoms_data_file> <attenuation_correction_factors_file> <path_to_normalization_and_attenuation_files> <normalization_file> <attenuation_image_file> <output_prefix> <non-interactive_mode>")
        sys.exit(1)
    raw_data_file = args[0]
    randoms_data_file = args[1]
    attenuation_correction_factors_file = args[2]
    path_to_normalization_and_attenuation_files = args[3]
    normalization_file = args[4]
    attenuation_image_file = args[5]
    output_prefix = args[6]
    non_interactive_mode = args[7] == "True"

    # Set up scatter estimator
    scatter_estimator = ScatterEstimator()
    scatter_estimator.set_collimator_file(path_to_normalization_and_attenuation_files)
    scatter_estimator.set_detector_file(path_to_normalization_and_attenuation_files)
    scatter_estimator.set_parameter_file(path_to_normalization_and_attenuation_files)
    scatter_estimator.read_from_file(raw_data_file)
    scatter_estimator.set_attenuation_correction_factors(attenuation_correction_factors_file)
    scatter_estimator.label_and_name(label_and_name)
    scatter_estimator.name_and_parameters(name_and_parameters)
    scatter_estimator.set_working_folder_file_overwrite(path_to_normalization_and_attenuation_files)
    scatter_estimator.set_attenuation_image(attenuation_image_file)
    scatter_estimator.process()
    scatter_estimator.normalise_zero_and_one(normalization_file)
    scatter_estimator.get_attenuation_image(get_attenuation_image)
    scatter_estimator.set_attenuation_type(get_attenuation_type)
    scatter_estimator.estimate_randoms(randoms_data_file)
    scatter_estimator.set_output_prefix(output_prefix)

    # Perform scatter estimation
    try:
        scatter_estimator.process()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Display scatter estimate and plot sinogram profile
    if not non_interactive_mode:
        print(f"Scatter estimate: {scatter_estimator.get_scatter_estimate()}")
        scatter_estimator.plot_sinogram_profile()

if __name__ == "__main__":
    main()