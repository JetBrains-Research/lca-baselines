  import logging
import numpy as np
import h5py
from typing import List, Dict
from pathlib import Path
from affinities.pipeline import Pipeline
from affinities.pipeline.request import BatchRequest
from affinities.pipeline.augmentation import (
    test_up_and_downsample,
    shift_and_crop,
    test_jitter_and_random,
    AddAffinities,
    BatchRequestError,
    test_shift_and_crop2,
    test_mirror_and_transpose,
    test_shift_and_crop1,
    test_shift_and_crop4,
    test_shift_and_crop3,
    SpecifiedLocation,
    PrintProfilingStats,
    __read_file,
    Batch,
    _add_to_batch,
    AddNonsymmetricAffinities,
    PipelineRequestError,
    test_shift_and_crop_static,
)

logging.basicConfig(level=logging.INFO)

def train_model(num_iterations: int):
    # Set up logging
    logger = logging.getLogger(__name__)

    # Import necessary libraries
    import numpy as np
    import h5py
    from typing import List, Dict
    from pathlib import Path
    from affinities.pipeline import Pipeline
    from affinities.pipeline.request import BatchRequest
    from affinities.pipeline.augmentation import (
        test_up_and_downsample,
        shift_and_crop,
        test_jitter_and_random,
        AddAffinities,
        BatchRequestError,
        test_shift_and_crop2,
        test_mirror_and_transpose,
        test_shift_and_crop1,
        test_shift_and_crop4,
        test_shift_and_crop3,
        SpecifiedLocation,
        PrintProfilingStats,
        __read_file,
        Batch,
        _add_to_batch,
        AddNonsymmetricAffinities,
        PipelineRequestError,
        test_shift_and_crop_static,
    )

    # Set up pipeline
    pipeline = Pipeline()

    # Set up batch request
    batch_request = BatchRequest(
        raw_intensities=True,
        labelled_objects=True,
        per_voxel_affinities=True,
        loss_weights=True,
        predicted_affinities=True,
        gradients_of_loss_with_respect_to_predicted_affinities=True,
    )

    # Set up configuration file
    config_file = "config.yaml"

    # Read configuration file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Calculate input and output sizes in world units
    input_size = config["input_size"]
    output_size = config["output_size"]

    # Formulate request for what a batch should contain
    batch_request.raw_intensities = (input_size, input_size, input_size)
    batch_request.labelled_objects = (output_size, output_size, output_size)
    batch_request.per_voxel_affinities = (output_size, output_size, output_size)
    batch_request.loss_weights = (output_size, output_size, output_size)
    batch_request.predicted_affinities = (output_size, output_size, output_size)
    batch_request.gradients_of_loss_with_respect_to_predicted_affinities = (
        output_size,
        output_size,
        output_size,
    )

    # Formulate snapshot request for inspection
    snapshot_request = BatchRequest(
        raw_intensities=True,
        labelled_objects=True,
        per_voxel_affinities=True,
        loss_weights=True,
        predicted_affinities=True,
        gradients_of_loss_with_respect_to_predicted_affinities=True,
    )

    # Assemble training pipeline
    pipeline.add_step(
        "read_batch",
        __read_file,
        batch_request=batch_request,
        snapshot_request=snapshot_request,
    )
    pipeline.add_step("normalize_raw_data", np.divide, out=batch_request.raw_intensities)
    pipeline.add_step("choose_random_location", np.random.randint, size=(1,))
    pipeline.add_step("apply_augmentations", test_up_and_downsample)
    pipeline.add_step("grow_boundary", shift_and_crop)
    pipeline.add_step("convert_labels_to_affinities", AddAffinities)
    pipeline.add_step("balance_labels", test_jitter_and_random)
    pipeline.add_step("pre_cache_batches", _add_to_batch)
    pipeline.add_step("perform_training_iteration", np.add)
    pipeline.add_step("save_batch_for_inspection", __read_file)
    pipeline.add_step("print_profiling_stats", PrintProfilingStats)

    # Print statement indicating start of training
    logger.info("Starting training...")

    # Build pipeline
    pipeline.build()

    # Request batches for specified number of iterations
    for i in range(num_iterations):
        pipeline.request_batch()

    # Print statement indicating end of training
    logger.info("Training complete.")