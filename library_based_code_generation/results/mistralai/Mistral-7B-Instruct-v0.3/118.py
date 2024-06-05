 ```python
from psychclone.transformations import *
from psychclone.constants import *

redundant_computation = True
async_halo_exchange = True
openmp_colouring = True
intrinsic_inlining = True

def transform_psy(psy):
    try:
        if redundant_computation:
            redundant_computation_pass(psy)
        if async_halo_exchange:
            async_halo_exchange_pass(psy)
        if openmp_colouring:
            openmp_colouring_pass(psy)
        if intrinsic_inlining:
            intrinsic_inlining_pass(psy)
        return psy
    except TransformationError as e:
        print(f"TransformationError: {e}")
        return psy
```