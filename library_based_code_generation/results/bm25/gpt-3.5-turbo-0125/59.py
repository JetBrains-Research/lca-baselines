import logging
import h5py

def train_model(num_iterations):
    raw_intensities = []
    labelled_objects = []
    per_voxel_affinities = []
    loss_weights = []
    predicted_affinities = []
    gradients = []

    # Read configuration file and calculate input/output sizes
    # Formulate batch request and snapshot request

    # Assemble training pipeline
    # - Read batches from HDF5 file
    # - Normalize raw data
    # - Choose random location for each batch
    # - Apply augmentations
    # - Grow boundary between labels
    # - Convert labels into affinities
    # - Balance labels
    # - Pre-cache batches
    # - Perform one training iteration for each batch
    # - Save batch as HDF5 file for inspection
    # - Print profiling stats

    print("Training started")
    # Build pipeline
    # Request batches for specified number of iterations
    print("Training ended")

if __name__ == "__main__":
    train_model(100)