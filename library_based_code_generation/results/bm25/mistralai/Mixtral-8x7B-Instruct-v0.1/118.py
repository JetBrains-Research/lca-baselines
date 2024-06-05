 from psyclone.psy import psy
from psyclone.transformations import (
    FunctionSpaceTo, FunctionSpaceFrom, IndexOfObject,
    InsertOclWriteToDeviceFunction, ObjectIndex,
    TestValidateWithAFunctionCall, TestApplyFunction,
    TestValidReturnObjectFromName, TestValidReturnObjectFromNumber,
    NodesToCodeBlock, PrepareCodeToInline, TeardownFunction,
    FunctionSpaces, FunctionFormat, FunctionSpace, BasisFunction,
    SomeFunction, TransformationError
)

apply_transformations = True
apply_redundant_computation = False
apply_async_halo_exchanges = False
apply_openmp_colouring = False
apply_intrinsic_inlining = False

def transform_psy(psy_obj):
    try:
        for invoke in psy_obj.invokes():
            if apply_transformations:
                function_space_to = FunctionSpaceTo(
                    function_space=FunctionSpace(
                        basis_functions=[
                            BasisFunction(
                                distribution_order=1,
                                distribution_method='block',
                                tile_size=1
                            )]
                    ),
                    target_function_name='some_function'
                )
                invoke.apply_transformations([function_space_to])

                index_of_object_trans = IndexOfObject(
                    object_name='some_variable',
                    object_index=ObjectIndex(0)
                )
                invoke.apply_transformations([index_of_object_trans])

                if apply_redundant_computation:
                    insert_ocl_write_to_device_function = InsertOclWriteToDeviceFunction(
                        function_name='some_function',
                        variable_name='some_variable',
                        kernel_name='redundant_computation'
                    )
                    invoke.apply_transformations(
                        [insert_ocl_write_to_device_function])

                if apply_async_halo_exchanges:
                    function_space_from = FunctionSpaceFrom(
                        function_space=FunctionSpace(
                            basis_functions=[
                                BasisFunction(
                                    distribution_order=1,
                                    distribution_method='block',
                                    tile_size=1
                                )]
                        ),
                        target_function_name='some_function'
                    )
                    invoke.apply_transformations([function_space_from])

                if apply_openmp_colouring:
                    test_validate_with_a_function_call = TestValidateWithAFunctionCall(
                        function_name='some_function',
                        kernel_name='openmp_colouring'
                    )
                    invoke.apply_transformations(
                        [test_validate_with_a_function_call])

                if apply_intrinsic_inlining:
                    test_apply_function = TestApplyFunction(
                        function_name='some_function',
                        kernel_name='intrinsic_inlining'
                    )
                    invoke.apply_transformations(
                        [test_apply_function])

                    nodes_to_code_block = NodesToCodeBlock(
                        node_names=[
                            'some_function_call_node'
                        ],
                        kernel_name='intrinsic_inlining'
                    )
                    invoke.apply_transformations(
                        [nodes_to_code_block])

                    prepare_code_to_inline = PrepareCodeToInline(
                        kernel_name='intrinsic_inlining'
                    )
                    invoke.apply_transformations(
                        [prepare_code_to_inline])

                    teardown_function = TeardownFunction(
                        kernel_name='intrinsic_inlining'
                    )
                    invoke.apply_transformations(
                        [teardown_function])

                function_spaces = FunctionSpaces(
                    function_format=FunctionFormat.OPENMP,
                    function_name='some_function'
                )
                invoke.apply_transformations([function_spaces])

                test_valid_return_object_from_name = TestValidReturnObjectFromName(
                    function_name='some_function',
                    variable_name='some_variable',
                    kernel_name='return_object_from_name'
                )
                invoke.apply_transformations(
                    [test_valid_return_object_from_name])

                test_valid_return_object_from_number = TestValidReturnObjectFromNumber(
                    function_name='some_function',
                    variable_index=0,
                    kernel_name='return_object_from_number'
                )
                invoke.apply_transformations(
                    [test_valid_return_object_from_number])

        return psy_obj

    except TransformationError as e:
        print(f'Error applying transformations: {e}')
        return None