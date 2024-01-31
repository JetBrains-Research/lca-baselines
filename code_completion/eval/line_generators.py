import argparse
from typing import Dict

import jsonlines
from evaluate import load
from codegen_metrics import bleu, chrf, codebleu, meteor, rougel, ruby
from thefuzz import fuzz
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from dataclasses import dataclass
from tqdm.auto import tqdm
import torch
import numpy as np
import random
import os
import logging

from code_completion.data_classes.datapoint_base import DatapointBase
from code_completion.data_classes.datapoint_commit_dataset import DatapointCommitDataset
from code_completion.model_hub.model_inference import get_input_data, get_model


@dataclass
class GeneratorConfig:
    input_data_path: str
    seq_max_len: int
    context_max: int
    model: str
    device: str
    best_perplexity: float
    tokenizer_path: str
    composer: str
    seed: int
    results_path: str


logging.basicConfig(level=logging.ERROR)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


@dataclass
class GenerationResults:
    prediction: list[str]
    gt: list[str]

    def append_result(self, prediction, gt):
        self.prediction.append(prediction)
        self.gt.append(gt)


class LineGeneratorBase:
    def __init__(self, model, device, max_seq_len, results_path):
        self.model = model
        self.device = device
        self.max_seq_len = max_seq_len
        self.results_path = results_path
        self.generation_results: Dict[str, GenerationResults] = dict()

    def choose_lines(self, datapoint) -> list[int]:
        raise NotImplementedError

    @staticmethod
    def _get_context(datapoint: DatapointBase, line_num: int) -> (str, str):
        """Method returns context and a line to predict"""
        context = "\n".join([datapoint.context] + [datapoint.get_prefix(line_num)])
        gt_line = datapoint.get_line(line_num)
        return context, gt_line

    @staticmethod
    def _get_zero_context(datapoint, line_num) -> (str, str):
        """Method returns context and a line to predict"""
        context = datapoint.get_prefix(line_num)
        gt_line = datapoint.get_line(line_num)
        return context, gt_line


    def generate_line(self, datapoint):
        raise NotImplementedError

    def calculate_exact_match(self):
        raise NotImplementedError

    def _load_tokenizer(self):
        raise NotImplementedError

    def tokenize(self, text):
        raise NotImplementedError

    def decode(self, generated_token_ids):
        raise NotImplementedError

    def _get_generation_config(self):
        raise NotImplementedError

    # @staticmethod
    # def _get_completion_lines(datapoint):
    #     return datapoint['completion'].split("\n")

    def aggregate_metric(self, metric_result):
        agg_result = 0.
        agg_len = 0
        metric_name = None
        for sc_name, sc_score in metric_result.items():
            agg_result += list(sc_score.values())[0] * len(self.generation_results[sc_name].gt)
            agg_len += len(self.generation_results[sc_name].gt)
            metric_name = list(sc_score.keys())[0]
        if len(metric_result) > 0:
            return {metric_name: agg_result / agg_len}

    def save_results(self, results):
        with jsonlines.open(self.results_path, 'a') as writer:
            writer.write(results)

# class RandomLineGenerator(LineGeneratorBase):
#
#     def choose_lines(self, datapoint) -> Dict[str, list[int]]:
#         datapoint['completion_lines'] = self._get_completion_lines(datapoint)
#         number_of_lines = len(datapoint['completion_lines'])
#         result = list()
#         while_count = 0
#         # result = [10 + ((i+1)*(number_of_lines-15)) // 5 for i in range(5)]
#         while len(result) < 20 and len(result) < 0.1 * number_of_lines and while_count<100:
#             line_num = random.randint(0, number_of_lines-1)
#             if line_num not in result and line_num > 10:
#                 line = datapoint['completion_lines'][line_num]
#                 line = line.strip()
#                 if self._line_condition_(line):
#                     result.append(line_num)
#             while_count += 1
#         return {'random': sorted(result)}
#
#     def _line_condition_(self, line):
#         if 5 < len(line) < 100:  # line length condition
#             if not self._contains_comment(line):  # filter lines with comments
#                 if 'print' not in line and 'import' not in line:  # filter printing and importing
#                     return True
#
#         return False
#
#
#     @staticmethod
#     def _contains_comment(line):
#         return '#' in line


class SpecificLineGenerator(LineGeneratorBase):
    @staticmethod
    def load_lines(datapoint: DatapointBase) -> dict[str, list[int]]:
        return datapoint.completion_lines

    @staticmethod
    def sample_noninformative(non_informative_lines: list[int], sample_size: int = 6, seed: int = 42):
        local_random = random.Random(seed)
        local_sample_size = min(len(non_informative_lines), sample_size)
        return local_random.sample(non_informative_lines, local_sample_size)

    #TODO: Rethink all the functionality below, since it was performed during data collection
    #
    # def choose_lines(self, datapoint) -> Dict[str, list[int]]:
    #     # datapoint['completion_lines'] = self._get_completion_lines(datapoint)
    #     number_of_lines = len(datapoint['completion_lines'])
    #     apis = self._collect_inproject_api(datapoint)
    #     apis['common_api'] = self._collect_common_api()
    #
    #     classified_lines = self._classify_lines(datapoint['completion_lines'], apis)
    #     non_informative_lines = self._get_noninformative_lines(datapoint['completion_lines'])
    #
    #     classified_lines = {k: list(set(v) - set(non_informative_lines)) for k, v in classified_lines.items()}
    #     classified_lines['non_informative'] = non_informative_lines
    #     blocked_lines = non_informative_lines + [idx for sc_lines in classified_lines.values() for idx in sc_lines]
    #     random_lines = self._select_lines_randomly(datapoint['completion'], blocked_lines)
    #     classified_lines['random'] = random_lines
    #     # print(classified_lines)
    #     return classified_lines
    #
    #
    # @staticmethod
    # def _select_lines_randomly(code: str, blocked_lines: list[int], num: int = 5):
    #     lines = code.split('\n')
    #     clean_code = ParsedFile(code=code).clean_comments()
    #     clean_lines = list(set(clean_code.split('\n')))
    #     clean_to_original = dict()
    #     for idx, clean_line in enumerate(clean_lines):
    #         mapping_lines = [or_idx for or_idx, or_line in enumerate(lines) if or_line == clean_line]
    #         if len(mapping_lines) > 0:
    #             clean_to_original[idx] = mapping_lines
    #
    #     count = 0
    #     random_lines = list()
    #     while len(random_lines) <= num and count < num * 100:
    #         clean_line_choice = random.choice(list(range(len(clean_lines))))
    #         if clean_line_choice in clean_to_original:
    #             original_line_choice = random.choice(clean_to_original[clean_line_choice])
    #             if original_line_choice not in random_lines and original_line_choice not in blocked_lines:
    #                 random_lines.append(original_line_choice)
    #         count += 1
    #     return random_lines
    #
    # @staticmethod
    # def _collect_inproject_api(datapoint):
    #     code_file = datapoint['completion']
    #     code_project = datapoint['context']
    #     pf_file = ParsedFile(code=code_file)
    #     pf_project = ParsedFile(code=code_project)
    #     infile_api = pf_file.function_names | pf_file.class_names
    #     inproject_api = pf_project.function_names | pf_project.class_names - infile_api
    #     return {'infile_api': [el.decode(pf_file.encoding) for el in infile_api],
    #             'inproject_api': [el.decode(pf_project.encoding) for el in inproject_api]}
    #
    # @staticmethod
    # def _collect_common_api():
    #     return COMMON_APIS
    #
    # @staticmethod
    # def _get_line_condition(line, apis):
    #     common_api = set(apis['common_api'])
    #     infile_api = set(apis['infile_api']) - common_api
    #     inproject_api = set(apis['inproject_api']) - infile_api
    #     inproject_api = inproject_api - common_api
    #     condition = {
    #         'infile': any([el in line for el in infile_api]),
    #         'inproject': any([el in line for el in inproject_api]),
    #         'common': any([el in line for el in common_api]),
    #     }
    #     return condition
    #
    # @staticmethod
    # def _classify_line(line_condition):
    #     if line_condition['infile'] and not line_condition['inproject']:
    #         return 'infile'
    #     elif line_condition['inproject']:
    #         return 'inproject'
    #     elif line_condition['common']:
    #         return 'common'
    #     else:
    #         return None
    #
    # def _classify_lines(self, completion_lines, apis):
    #     conditions = list()
    #     for line in completion_lines:
    #         conditions.append(self._get_line_condition(line, apis))
    #     result = {
    #         'infile': list(), 'inproject': list(), 'common': list()
    #     }
    #     for num_line, c in enumerate(conditions):
    #         c_class = self._classify_line(c)
    #         if c_class:
    #             result[c_class].append(num_line)
    #     return result
    #
    # @staticmethod
    # def _get_noninformative_lines(completion_lines, warmup_len=10):
    #     ni_lines = list(range(warmup_len))
    #     for num_line, line in enumerate(completion_lines):
    #         if len(line) <= 5 or len(line) >= 150:
    #             ni_lines.append(num_line)
    #         elif 'print' in line or 'import' in line:
    #             ni_lines.append(num_line)
    #         elif line.strip().startswith('def ') or line.strip().startswith('class '):
    #             ni_lines.append(num_line)
    #
    #     return list(set(ni_lines))


class LineGeneratorHF(SpecificLineGenerator):
    def __init__(self, model, device, max_seq_len, results_path, tokenizer_path):
        super().__init__(model, device, max_seq_len, results_path)
        self.tokenizer_path = tokenizer_path
        self._tokenizer: AutoTokenizer
        self._load_tokenizer()

    @torch.inference_mode()
    def generate_line(self, datapoint: DatapointBase, use_zero_context: bool = False) -> dict[str, int]:
        # dict_of_lines = self.choose_lines(datapoint)
        dict_of_lines = self.load_lines(datapoint)
        # non_informative_lines = dict_of_lines.pop('non_informative', None)
        # if non_informative_lines is not None:
        #     dict_of_lines['non_informative'] = self.sample_noninformative(non_informative_lines)
        gen_config = self._get_generation_config()
        for sc_name, list_of_lines in dict_of_lines.items():
            # print('='*25, sc_name, '='*25)
            self.generation_results[sc_name] = GenerationResults(list(), list())
            for line_num in list_of_lines:
                context, gt_line = self._get_context(datapoint, line_num)
                if use_zero_context:
                    context, gt_line = self._get_zero_context(datapoint, line_num)
                # When the context is too long, we want to crop the beginning for more efficient tokenization
                if len(context) > self.max_seq_len * 6:
                    context = context[-self.max_seq_len * 6:]
                input_ids = self.tokenize(context)[..., -self.max_seq_len:]
                if input_ids.size(-1) < 1:
                    new_size = torch.Size(list(input_ids.size())[:-1] + [1])
                    input_ids = torch.full(new_size, self._tokenizer.bos_token_id)
                # print(input_ids.shape)
                input_ids = input_ids.to(self.device)
                out = self.model.generate(input_ids, **gen_config)
                out = out[..., input_ids.size(-1):]
                # print(out.size())
                prediction = self.decode(out)
                # prediction_line = prediction
                prediction = prediction.strip("\n")
                prediction_line = prediction.split("\n")[0]
                # print(context)
                # print('Prediction: ', prediction)  #prediction_line)
                # print('GT:', gt_line)
                # print('===---'*30)
                self.save_results({'original_prediction': prediction, 'prediction_line': prediction_line, 'ground_truth': gt_line, 'line_class': sc_name, 'zero_context': use_zero_context})
                self.generation_results[sc_name].append_result(prediction=prediction_line, gt=gt_line)

        # datapoint.pop('completion_lines', None)
        return {k: len(v) for k, v in dict_of_lines.items()}

    def _load_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def tokenize(self, text):
        return self._tokenizer(text, return_tensors='pt', padding=False)['input_ids']

    def _get_generation_config(self):
        class StopOnNewLine(StoppingCriteria):
            def __init__(self, tokenizer):
                self.stop_ids = set()
                for k, tok_id in tokenizer.vocab.items():
                    s = tokenizer.convert_tokens_to_string([k])
                    if '\n' in s:
                        self.stop_ids.add(tok_id)
                self._num_generated_tokens = 0

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                assert input_ids.shape[0] == 1  # only batch_size 1 is supported
                if self._num_generated_tokens < 5:
                    self._num_generated_tokens += 1
                    return False
                elif input_ids[0, -1].item() in self.stop_ids:
                    self._num_generated_tokens = 0
                    return True
                else:
                    self._num_generated_tokens += 1
                    return False

        stopping_criteria = StoppingCriteriaList([StopOnNewLine(self._tokenizer)])
        # newline_token_id = self._tokenizer.encode('\n', add_special_tokens=False)[0]
        return {
            'max_new_tokens': 100,
            'do_sample': False,
            'stopping_criteria': stopping_criteria,
            'eos_token_id': self._tokenizer.eos_token_id,
            'pad_token_id': self._tokenizer.eos_token_id,
        }

    def decode(self, generated_token_ids):
        return self._tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]

    def calculate_exact_match(self):
        exact_match = load("exact_match")
        results = dict()
        for sc_name, gen_res in self.generation_results.items():
            if len(gen_res.gt) > 0:
                results[sc_name] = exact_match.compute(
                    references=[item.strip() for item in gen_res.gt],
                    predictions=[item.strip() for item in gen_res.prediction],
                )
        return results

    def calculate_edit_similarity(self):
        similarity = 0.
        count = 0
        result = dict()
        for sc_name, gen_res in self.generation_results.items():
            for pred, gt in zip(gen_res.prediction, gen_res.gt):
                similarity += fuzz.ratio(pred, gt)
                count += 1
            if count > 0:
                result[sc_name] = {'edit_similarity': similarity / count}
        return result



@torch.inference_mode()
def evaluate_generation(args: GeneratorConfig):
    set_seed(args.seed)
    loaded_data = get_input_data(args)
    if isinstance(loaded_data[0], dict):
        input_data = [DatapointCommitDataset(**input_dict) for input_dict in loaded_data]
    elif isinstance(loaded_data[0], DatapointCommitDataset):
        input_data = loaded_data.copy()
    else:
        raise NotImplementedError

    model, device = get_model(args)

    def calculate_metrics(model=model, device=device, use_zero_context=False, args=args, input_data=input_data):
        em_dict = dict()
        es_dict = dict()
        em_dict['all'] = list()
        es_dict['all'] = list()
        sc_counts = None
        for datapoint in tqdm(input_data):
            generator = LineGeneratorHF(model, device, max_seq_len=args.seq_max_len, tokenizer_path=args.tokenizer_path, results_path=args.results_path)
            el_counts = generator.generate_line(datapoint, use_zero_context=use_zero_context)
            if sc_counts is None:
                sc_counts = el_counts
            else:
                for k in el_counts.keys():
                    sc_counts[k] += el_counts[k]
            em = generator.calculate_exact_match()
            es = generator.calculate_edit_similarity()
            em_dict['all'].append(generator.aggregate_metric(em)['exact_match'])
            es_dict['all'].append(generator.aggregate_metric(es)['edit_similarity'])
            for sc_name in em.keys():
                if sc_name not in em_dict:
                    em_dict[sc_name] = list()
                if sc_name not in es_dict:
                    es_dict[sc_name] = list()

                try:
                    em_dict[sc_name].append(em[sc_name]['exact_match'])
                except KeyError:
                    pass
                try:
                    es_dict[sc_name].append(es[sc_name]['edit_similarity'])
                except KeyError:
                    pass
        return em_dict, es_dict, sc_counts

    def process_results(use_zero_context):
        em_dict, es_dict, sc_counts = calculate_metrics(use_zero_context=use_zero_context)
        if use_zero_context:
            print(f'Final results for zero context:')
        else:
            print(f'Final results for full context:')
        for sc_name in em_dict.keys():
            em_list = em_dict[sc_name]
            es_list = es_dict[sc_name]
            print(f'Metrics for {sc_name} lines: EM {sum(em_list) / len(em_list):.2f}, ES {sum(es_list) / len(es_list):.2f}')

        return em_dict, es_dict, sc_counts

    set_seed(args.seed)
    em_dict_0, es_dict_0, line_counts_0 = process_results(use_zero_context=True)
    set_seed(args.seed)
    em_dict, es_dict, line_counts = process_results(use_zero_context=False)
    assert line_counts_0 == line_counts, "you have different line counts"
    em_diff_dict = dict()
    for sc_name in em_dict.keys():
        em_list = em_dict[sc_name]
        em_list_0 = em_dict_0[sc_name]
        assert len(em_list) == len(em_list_0), 'your score has different lengths'
        em_diff_dict[sc_name] = {
            'positive': sum([(sc - sc_0) > 0 for sc, sc_0 in zip(em_list, em_list_0)]) / len(em_list),
            'negative': sum([(sc - sc_0) < 0 for sc, sc_0 in zip(em_list, em_list_0)]) / len(em_list),
            'zero': sum([(sc - sc_0) == 0 for sc, sc_0 in zip(em_list, em_list_0)]) / len(em_list),
        }

    return [
        {
            'em_zero': {sc_name: sum(m_list) / len(m_list) for sc_name, m_list in em_dict_0.items()},
            'es_zero': {sc_name: sum(m_list) / len(m_list) for sc_name, m_list in es_dict_0.items()},
            'em': {sc_name: sum(m_list) / len(m_list) for sc_name, m_list in em_dict.items()},
            'es': {sc_name: sum(m_list) / len(m_list) for sc_name, m_list in es_dict.items()},
        },
        {
            'em_zero_list': em_dict_0,
            'es_zero_list': es_dict_0,
            'em_list': em_dict,
            'es_list': es_dict,
        },
        em_diff_dict,
        line_counts
    ]


    # print(f'Final results for zero context: '
    #       f'EM {sum(em_list) / len(em_list):.2f}, ES {sum(es_list) / len(es_list):.2f}')


if __name__ == '__main__':
    args = GeneratorConfig(
        input_data_path="/home/glukhov/long_code_arena/lca/data/python/smol/model_inputs_composer_path_distance.json",
        seq_max_len=3500 - 30,
        context_max=3500,
        model="starcoder1b",
        device="cuda",
        best_perplexity=0.,
        tokenizer_path="bigcode/starcoderbase-1b",
        composer="path_distance",
        seed=42
    )
    print(args.input_data_path)
    out = evaluate_generation(args)
    for out_ in out:
        print(out_)
        print()
