from composers.base_composer import ComposerBase
from data_classes.datapoint_base import DatapointBase


class OneCompletionFileComposer(ComposerBase):
    def __init__(self, **composer_args):
        super().__init__(**composer_args)

    def completion_composer(self, datapoint: DatapointBase) -> str:
        completion = datapoint.completion_dict
        assert len(completion) == 1, 'Only one file should be completed'
        content = list(completion.values())[0]
        return content
