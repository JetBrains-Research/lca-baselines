from psyclone.transformations import RedundantComputationTrans, \
    DynHaloExchangeTrans, OMPParallelTrans, InlineTrans
from psyclone import TransformationError

apply_transformations = True
apply_halo_exchange = True
apply_omp_colouring = True
apply_intrinsic_inlining = True

def apply_transformations_to_psy(psy):
    try:
        for invoke in psy.invokes.invoke_list:
            if apply_transformations:
                if apply_halo_exchange:
                    trans = DynHaloExchangeTrans()
                    trans.apply(invoke.schedule)
                if apply_omp_colouring:
                    trans = OMPParallelTrans()
                    trans.apply(invoke.schedule)
                if apply_intrinsic_inlining:
                    trans = InlineTrans()
                    trans.apply(invoke.schedule)
    except TransformationError as e:
        print("TransformationError: {0}".format(e))
    
    return psy