  from psyclone import *

redundant_computation = True
asynchronous_halo_exchanges = True
openmp_colouring = True
intrinsic_inlining = True

def transform_psy(psy):
    for invoke in psy.invokes:
        if redundant_computation:
            try:
                function_space_to(invoke, "redundant_computation")
            except TransformationError:
                pass
        if asynchronous_halo_exchanges:
            try:
                index_of_object(invoke, "asynchronous_halo_exchanges")
            except TransformationError:
                pass
        if openmp_colouring:
            try:
                _insert_ocl_write_to_device_function(invoke, "openmp_colouring")
            except TransformationError:
                pass
        if intrinsic_inlining:
            try:
                function_space_from(invoke, "intrinsic_inlining")
            except TransformationError:
                pass
    return psy

transformed_psy = transform_psy(psy)