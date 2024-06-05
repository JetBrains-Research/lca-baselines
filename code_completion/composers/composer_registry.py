COMPOSERS = {
    'none': None,
    'naive': {'module': 'composers.naive_composer', 'name': 'NaiveComposer'},
    'alphabetical': {'module': 'composers.alphabetical_composer', 'name': 'AlphabeticalComposer'},
    'path_distance': {'module': 'composers.path_distance_composer', 'name': 'PathDistanceComposer'},
    'file_length': {'module': 'composers.file_length_composer', 'name': 'FileLengthComposer'},
    'half_memory': {'module': 'composers.half_memory_composer', 'name': 'HalfMemoryComposer'},
    'half_memory_path_distance':
        {'module': 'composers.half_memory_path_distance_composer', 'name': 'HalfMemoryPathDistanceComposer'},
    'function_class_mask_half': {'module': 'composers.function_class_half_mask', 'name': 'FuncClassComposer'},
    'function_class_mask_one': {'module': 'composers.function_class_mask_one', 'name': 'FuncClassComposerOne'},
    'imports_first': {'module': 'composers.imports_first_composer', 'name': 'ImportsFirstComposer'},
}
