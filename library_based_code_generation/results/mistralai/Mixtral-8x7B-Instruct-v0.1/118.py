 ```python
from psyclone.psyir.transformations import (RedundantComputation, 
                                            AsynchronousHaloExchanges, 
                                            OpenMPColouring, Inlining)
from psyclone.errors import TransformationError

apply_transformations = True
reduce_redundant_computation = True
apply_async_halo_exchanges = True
apply_openmp_colouring = True
apply_inlining = True

def transform_psy(psy):
    global apply_transformations, reduce_redundant_computation, apply_async_halo_exchanges, \
           apply_openmp_colouring, apply_inlining

    try:
        if apply_transformations:
            for invoke in psy.invokes:
                if reduce_redundant_computation:
                    RedundantComputation().apply(invoke)
                if apply_async_halo_exchanges:
                    AsynchronousHaloExchanges().apply(invoke)
                if apply_openmp_colouring:
                    OpenMPColouring().apply(invoke)
                if apply_inlining:
                    Inlining().apply(invoke)
    except TransformationError as err:
        print(f"TransformationError: {err}")
        return None

    return psy
```