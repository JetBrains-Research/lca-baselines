from psyclone import transformations, constants

apply_redundant_computation = True
apply_async_halo_exchange = True
apply_openmp_colouring = True
apply_intrinsic_inlining = True

def apply_transformations(psy):
    try:
        for invoke in psy.invokes.invoke_list:
            if apply_redundant_computation:
                transformations.apply(invoke, ["redundant_computation"])
            if apply_async_halo_exchange:
                transformations.apply(invoke, ["async_halo_exchange"])
            if apply_openmp_colouring:
                transformations.apply(invoke, ["openmp_colour"])
            if apply_intrinsic_inlining:
                transformations.apply(invoke, ["intrinsic_inline"])
    except transformations.TransformationError as e:
        print("TransformationError: {0}".format(e))
    return psy