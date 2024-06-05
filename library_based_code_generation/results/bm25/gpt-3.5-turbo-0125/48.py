import sys
import stir
from stir import *
from stir import label_and_name, name_and_parameters, PoissonLogLikelihoodWithLinearModelForMeanAndProjData, normalise_zero_and_one, estimate_randoms, from_acquisition_data, set_randoms, ListmodeToSinograms, TestSTIRAcquisitionDataAlgebraFile, AcquisitionData, read_from_file, set_acquisition_data, DataHandleVector, preprocess_acquisition_data, new_acquisition_data, AcquisitionDataProcessor, set_parameter_file, set_collimator_file, set_detector_file, petmr_data_path

def main():
    try:
        # Handle command-line options for file paths and other parameters
        
        # Set up listmode-to-sinograms converter object
        converter = ListmodeToSinograms()
        converter.set_input_listmode_filename("listmode_file")
        converter.set_output_prefix("sinograms_file_prefix")
        converter.set_template("raw_data_template")
        
        # Set time interval and flags for storing delayed coincidences
        converter.set_time_interval("scanning_time_interval")
        converter.set_store_delayed_flag(True)
        
        # Process data and estimate randoms
        converter.process()
        sinograms = converter.get_output()
        estimated_randoms = estimate_randoms(sinograms)
        
        # Write estimated randoms to a file
        estimated_randoms.write_to_file("randoms_file")
        
        # Copy acquisition data into Python arrays
        acq_data = from_acquisition_data(sinograms)
        print("Acquisition data dimensions:", acq_data.dimensions())
        print("Total number of delayed coincidences:", acq_data.sum())
        print("Estimated randoms:", estimated_randoms.sum())
        print("Max values:", acq_data.max())
        
        # Display sinogram if not in non-interactive mode
        if not non_interactive_mode:
            sinograms.display()
    
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()