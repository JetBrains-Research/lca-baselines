import logging
import numpy as np
import h5py

def train_model(num_iterations):
    raw_intensities = np.array([])
    labelled_objects = np.array([])
    per_voxel_affinities = np.array([])
    loss_weights = np.array([])
    predicted_affinities = np.array([])
    gradients = np.array([])
    
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
    # - Save passing batch as HDF5 file for inspection
    # - Print profiling stats
    
    logging.info("Start of training")
    
    # Build pipeline
    
    # Request batches for specified number of iterations
    
    logging.info("End of training")

if __name__ == "__main__":
    train_model(100)