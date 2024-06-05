```python
import sys
import argparse
import numpy as np
from sirf.STIR import AcquisitionData, ListmodeToSinograms, read_from_file, PoissonLogLikelihoodWithLinearModelForMeanAndProjData, estimate_randoms

def main():
    parser = argparse.ArgumentParser(description="Estimate randoms from listmode file and compare with original delayed coincidences.")
    parser.add_argument("--listmode_file", required=True, help="Path to the listmode file")
    parser.add_argument("--sinograms_prefix", required=True, help="Prefix for the sinograms file")
    parser.add_argument("--randoms_file", required=True, help="Path to the randoms file")
    parser.add_argument("--raw_data_template", required=True, help="Path to the raw data template")
    parser.add_argument("--scanning_time_interval", type=int, nargs=2, required=True, help="Scanning time interval as two integers (start, end)")
    parser.add_argument("--reconstruction_engine", required=True, help="Reconstruction engine to use")
    parser.add_argument("--acquisition_data_storage_scheme", required=True, help="Acquisition data storage scheme")
    parser.add_argument("--non_interactive", action='store_true', help="Run in non-interactive mode")
    
    args = parser.parse_args()

    try:
        AcquisitionData.set_storage_scheme(args.acquisition_data_storage_scheme)
        
        # Set up the listmode to sinograms converter
        lm2sino = ListmodeToSinograms()
        lm2sino.set_input(args.listmode_file)
        lm2sino.set_output_prefix(args.sinograms_prefix)
        lm2sino.set_template(args.raw_data_template)
        lm2sino.set_time_interval(args.scanning_time_interval[0], args.scanning_time_interval[1])
        lm2sino.set_flag("store_delayeds", True)
        
        # Process the data
        lm2sino.process()
        
        # Get access to the sinograms
        sinograms = lm2sino.get_output()
        
        # Estimate randoms
        randoms = estimate_randoms(sinograms)
        
        # Write estimated randoms to a file
        randoms.write(args.randoms_file)
        
        # Copy acquisition data into Python arrays
        acq_data = sinograms.as_array()
        randoms_data = randoms.as_array()
        
        # Print out information
        print(f"Acquisition data dimensions: {acq_data.shape}")
        print(f"Total number of delayed coincidences: {np.sum(acq_data)}")
        print(f"Total number of estimated randoms: {np.sum(randoms_data)}")
        print(f"Max value in acquisition data: {np.max(acq_data)}")
        print(f"Max value in estimated randoms: {np.max(randoms_data)}")
        
        if not args.non_interactive:
            # Display a single sinogram
            import matplotlib.pyplot as plt
            plt.imshow(acq_data[0], cmap='viridis')
            plt.title("Single Sinogram")
            plt.colorbar()
            plt.show()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```