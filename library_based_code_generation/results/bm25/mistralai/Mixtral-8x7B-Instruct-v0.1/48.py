 import argparse
import sys
from functools import partial

import stir.recon.emrecon_classes
import stir.data
import stir.util
import stir.util.math
from stir.data import Sinogram, AcquisitionData, DataHandleVector
from stir.recon.emrecon_classes import (
    ListmodeToSinograms,
    PoissonLogLikelihoodWithLinearModelForMeanAndProjData,
)
from stir.util import name_and_parameters, label_and_name
from stir.util.math import normalise_zero_and_one

def process_data(
    listmode_file,
    sinograms_file_prefix,
    randoms_file,
    raw_data_template,
    scanning_time_interval,
    reconstruction_engine,
    acquisition_data_storage_scheme,
    non_interactive,
):
    # Create ListmodeToSinograms object
    listmode_to_sinograms = ListmodeToSinograms(
        list_mode_file_name=listmode_file,
        sinogram_file_name_prefix=sinograms_file_prefix,
        raw_data_template=raw_data_template,
        scanning_time_interval=scanning_time_interval,
        reconstruction_engine=reconstruction_engine,
        acquisition_data_storage_scheme=acquisition_data_storage_scheme,
    )

    # Process data
    listmode_to_sinograms.process_listmode_data()

    # Get sinograms
    sinograms = listmode_to_sinograms.get_sinograms()

    # Estimate randoms
    estimated_randoms = estimate_randoms(sinograms)

    # Write estimated randoms to file
    estimated_randoms.write_to_file(randoms_file)

    # Copy acquisition data into Python arrays
    acquisition_data = from_acquisition_data(listmode_to_sinograms.get_acquisition_data())

    # Print acquisition data dimensions, total number of delayed coincidences and estimated randoms, and max values
    print(
        f"Acquisition data dimensions: {acquisition_data.get_data_handles().get_n_elements()}"
    )
    print(
        f"Total number of delayed coincidences: {listmode_to_sinograms.get_num_delayed_coincidences()}"
    )
    print(
        f"Total number of estimated randoms: {estimated_randoms.get_data_handles().get_n_elements()}"
    )
    print(f"Max value in acquisition data: {acquisition_data.get_max_value()}")

    # Display a single sinogram if not in non-interactive mode
    if not non_interactive:
        sinograms[0].display()

def estimate_randoms(sinograms):
    # Set up Maximum Likelihood estimation for randoms
    likelihood_model = PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
    likelihood_model.set_up(sinograms)

    # Normalise sinograms
    normalised_sinograms = normalise_zero_and_one(sinograms)

    # Estimate randoms
    estimated_randoms = likelihood_model.estimate_randoms(normalised_sinograms)

    return estimated_randoms

def main(args):
    # Parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("listmode_file", help="Path to listmode file")
    parser.add_argument(
        "sinograms_file_prefix", help="Prefix for sinograms file names"
    )
    parser.add_argument("randoms_file", help="Path to randoms file"
```