import dataclasses
import json
import multiprocessing
import os.path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable

import omegaconf
from datasets import load_dataset
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from data_classes.datapoint_base import DatapointBase
from data_classes.datapoint_commit_dataset import DatapointCommitDataset


@dataclass
class TokenizerOutput:
    context: List[int]
    completion: List[int]


class PreprocessorBase:
    def __init__(self,
                 dataset_params: str | dict,
                 tokenizer_path: str | None = None,
                 context_len_char: int = 60_000,
                 context_composer: Callable[[Dict[str, Any]], str] | None = None,
                 completion_composer: Callable[[Dict[str, Any]], str] | None = None,
                 data_source: str = 'hf',
                 ):
        self.dataset_params = dataset_params
        self.data: list[DatapointBase] = self._load_data(dataset_params)
        self.prepared_data: Optional[List[Dict[str, Any]]] = None
        self.tokenizer_path = tokenizer_path
        self.context_composer = context_composer
        self.completion_composer = completion_composer
        self.data_source = data_source
        self.context_len_char = context_len_char

    def compose_context(self, context: Dict[str, str]) -> str:
        raise NotImplementedError

    def compose_completion(self, context: Dict[str, str]) -> str:
        raise NotImplementedError

    def prepare_data(self):
        print('Data Preparation...')
        self.prepared_data = list()
        for datapoint in tqdm(self.data):
            new_datapoint = dict()
            new_datapoint['repo_id'] = datapoint.repo_id
            new_datapoint['repo_name'] = datapoint.repo_name
            new_datapoint['completion_lines'] = datapoint.completion_lines

            if self.context_composer is None:
                new_datapoint['context'] = self.compose_context(datapoint)
            else:
                new_datapoint['context'] = self.context_composer(datapoint)
            if self.completion_composer is None:
                new_datapoint['completion'] = self.compose_completion(datapoint)
            else:
                new_datapoint['completion'] = self.completion_composer(datapoint)

            # Following fields must be filled after tokenization
            new_datapoint['context_len'] = None  # number of tokens in `context`
            new_datapoint['model_input'] = None  # tokenized `context` + tokenized `completion`
            # new_datapoint['common_api'] = common_api
            self.prepared_data.append(type(datapoint)(**new_datapoint))

    def _datapoint_to_model_input(self, datapoint: DatapointBase) -> DatapointBase:
        datapoint = datapoint.to_model_input(self.tokenize_datapoint)
        return datapoint

    def prepare_model_input_parallel(self, num_workers=1, dataset_path=None):
        self.prepare_data()
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        if os.path.exists(dataset_path):
            os.remove(dataset_path)

        list_to_save = list()

        print('Tokenization...')
        if num_workers == 1:
            result = [self._datapoint_to_model_input(datapoint) for datapoint in tqdm(self.prepared_data)]
            for p in result:
                list_to_save.append(dataclasses.asdict(p))
        else:
            with Parallel(num_workers) as pool:
                result = pool(delayed(self._datapoint_to_model_input)(datapoint) for datapoint in self.prepared_data)
            for p in tqdm(result):
                list_to_save.append(dataclasses.asdict(p))

        with open(dataset_path, 'w') as json_file:
            json.dump(list_to_save, json_file)


    def tokenize(self, text) -> List[int]:
        raise NotImplementedError

    def tokenize_datapoint(self, datapoint: DatapointBase) -> TokenizerOutput:
        # print(len(datapoint.context), len(datapoint.completion))
        chunk_size = 1000  # size in lines
        cropped_context = datapoint.context[-self.context_len_char:]  # TODO: connect this to max_seq_len
        # context_lines = cropped_context.split('\n')
        # context_chunks_by_lines = [context_lines[i:i+chunk_size] for i in range(len(context_lines)//chunk_size)]
        # context_chunks = ['\n'.join(lines_chunk) for lines_chunk in context_chunks_by_lines]
        # splitting_char = 'METASEP\n'
        # context_chunks = cropped_context.split(splitting_char)
        # context_chunks[:-1] = [chunk + splitting_char for chunk in context_chunks[:-1]]
        context_chunks = [cropped_context]
        whitespace_char = '\n\n'
        context_chunks = [new_chunk for chunk in context_chunks for new_chunk in chunk.split(whitespace_char)]
        context_chunks[:-1] = [chunk + whitespace_char for chunk in context_chunks[:-1]]
        tokenized_chunks = [self.tokenize(chunk) for chunk in context_chunks]
        return TokenizerOutput(
            # context=self.tokenize(cropped_context),
            context=[token_id for chunk in tokenized_chunks for token_id in chunk],
            completion=self.tokenize(datapoint.completion)
        )

    def save_model_inputs(self, filepath='lca/code_generation/data/model_inputs.json'):
        with open(filepath, 'w') as f:
            json.dump(self.prepared_data, f)

    def _load_data(self, dataset_params: str | dict) -> list[DatapointBase]:
        if True: #self.data_source == 'hf':
            data = list()
            if isinstance(dataset_params, str):
                hf_data = load_dataset(dataset_params, split='test')
            elif isinstance(dataset_params, omegaconf.dictconfig.DictConfig):
                hf_data = load_dataset(split='test', **dataset_params)
            else:
                raise ValueError('check `config.dataset`, it must be string or dictionary')

            repos_list = list(set([hf_dp['repo'] for hf_dp in hf_data]))
            repos_map = {repo: repo_num for repo_num, repo in enumerate(repos_list)}

            for hf_dp in hf_data:
                dp = dict()
                dp['repo_name'] = hf_dp['repo']
                dp['repo_id'] = repos_map[hf_dp['repo']]
                dp['completion_lines'] = hf_dp['completion_lines']
                # dp['completion_lines_raw'] = hf_dp['completion_lines_raw']
                filenames, contents = hf_dp['repo_snapshot']['filename'], hf_dp['repo_snapshot']['content']
                assert len(filenames) == len(contents)
                dp['context_dict'] = {filename: content for filename, content in zip(filenames, contents)}
                # dp['context_dict'] = {el['filename']: el['content'] for el in hf_dp['repo_snapshot']}
                dp['completion_dict'] = {hf_dp['completion_file']['filename']: hf_dp['completion_file']['content']}
                data.append(DatapointCommitDataset(**dp))

            return data

        # with open(path, 'r') as f:
        #     data = json.load(f)
        # return data


from transformers import AutoTokenizer
class HFPreprocessor(PreprocessorBase):
    def __init__(self, dataset_params, tokenizer_path, context_len_char=60_000, **composers):
        super().__init__(dataset_params, tokenizer_path, context_len_char, **composers)
        self.lang_sep_symbol = ''
        self.meta_info_sep_symbol = 'METASEP'
        self.extension = ''
        self._tokenizer: AutoTokenizer
        self._load_tokenizer(self.tokenizer_path)

    def compose_context(self, datapoint: DatapointBase) -> str:
        context = datapoint.context_dict
        repo_name = datapoint.repo_name
        # You could implement specific order of contents in composed_content
        composed_content = [path + self.meta_info_sep_symbol + content for path, content in context.items()]
        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"
        return repo_metainfo + self.lang_sep_symbol.join(composed_content)

    def compose_completion(self, datapoint: DatapointBase) -> str:
        completion = datapoint.completion_dict
        # TODO: move path to the context
        composed_content = [path + self.meta_info_sep_symbol + content for path, content in completion.items()]
        return self.lang_sep_symbol + self.lang_sep_symbol.join(composed_content)

    def tokenize(self, text) -> List[int]:
        return self._tokenizer(text, add_special_tokens=False)['input_ids']

    def _load_tokenizer(self, path):
        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)


class StarcoderPreprocessor(HFPreprocessor):
    def __init__(self, dataset_params, tokenizer_path="bigcode/starcoder", context_len_char=60_000, **composers):
        super().__init__(dataset_params, tokenizer_path, context_len_char, **composers)
        self.lang_sep_symbol = 'LANGSEP'
        self.meta_info_sep_symbol = 'METASEP'
        self.extension = '.py'
        self._tokenizer: AutoTokenizer
