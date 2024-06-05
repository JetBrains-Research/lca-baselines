```python
from psyclone.transformations import (OMPParallelLoopTrans, 
                                      RedundantComputationTrans, 
                                      Dynamo0p3AsyncHaloExchangeTrans, 
                                      Dynamo0p3ColourTrans, 
                                      DynamoOMPParallelLoopTrans, 
                                      KernelModuleInlineTrans)
from psyclone.psyGen import TransformationError

# Define boolean variables to control the application of transformations
apply_redundant_computation = True
apply_async_halo_exchange = True
apply_omp_colouring = True
apply_intrinsic_inlining = True

def apply_transformations(psy):
    # Instantiate the transformations
    rc_trans = RedundantComputationTrans()
    async_halo_trans = Dynamo0p3AsyncHaloExchangeTrans()
    colour_trans = Dynamo0p3ColourTrans()
    omp_parallel_trans = DynamoOMPParallelLoopTrans()
    inline_trans = KernelModuleInlineTrans()

    for invoke in psy.invokes.invoke_list:
        schedule = invoke.schedule

        try:
            if apply_redundant_computation:
                for loop in schedule.loops():
                    rc_trans.apply(loop)

            if apply_async_halo_exchange:
                for loop in schedule.loops():
                    async_halo_trans.apply(loop)

            if apply_omp_colouring:
                for loop in schedule.loops():
                    colour_trans.apply(loop)
                    omp_parallel_trans.apply(loop)

            if apply_intrinsic_inlining:
                for kernel in schedule.coded_kernels():
                    inline_trans.apply(kernel)

        except TransformationError as error:
            print(f"Transformation error: {error}")

    return psy
```