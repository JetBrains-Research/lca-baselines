  from psyclone import transformations, constants

redundant_computation = True
asynchronous_halo_exchanges = True
openmp_colouring = True
intrinsic_inlining = True

def transform_psy(psy):
    for invoke in psy.invokes:
        if redundant_computation:
            try:
                transformations.redundant_computation(invoke)
            except TransformationError as e:
                print(f"Error applying redundant computation: {e}")
        if asynchronous_halo_exchanges:
            try:
                transformations.asynchronous_halo_exchanges(invoke)
            except TransformationError as e:
                print(f"Error applying asynchronous halo exchanges: {e}")
        if openmp_colouring:
            try:
                transformations.openmp_colouring(invoke)
            except TransformationError as e:
                print(f"Error applying OpenMP colouring: {e}")
        if intrinsic_inlining:
            try:
                transformations.intrinsic_inlining(invoke)
            except TransformationError as e:
                print(f"Error applying intrinsic inlining: {e}")
    return psy