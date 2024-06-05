import argparse
import importlib
import os

from composers.composer_registry import COMPOSERS

PREPROCESSORS = {
    'starcoder': {'module': 'eval.preprocessors', 'name': 'StarcoderPreprocessor'},
    'huggingface': {'module': 'eval.preprocessors', 'name': 'HFPreprocessor'},
}


def get_preprocessor(args):
    module = importlib.import_module(PREPROCESSORS[args.model]['module'])
    preprocessor = getattr(module, PREPROCESSORS[args.model]['name'])
    return preprocessor


def get_composers(args, composer_args):
    if COMPOSERS[args.composers] is not None:
        composer_module = importlib.import_module(COMPOSERS[args.composers]['module'])
        composer = getattr(composer_module, COMPOSERS[args.composers]['name'])(**composer_args)

    else:
        return {
            'context_composer': None,
            'completion_composer': None,
        }

    return {
        'context_composer': composer.context_composer,
        'completion_composer': composer.completion_composer
    }

def resolve_directories(args):
    if args.out_dir is not None:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        return args.out_dir
    path = os.path.join(os.getcwd(), 'data', f'{args.model}_inputs')
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def preprocess(args, composer_args):
    composers = get_composers(args, composer_args)
    prepared_dataset_path = os.path.join(resolve_directories(args), f'model_inputs_composer_{args.composers}.json')

    preprocessor = get_preprocessor(args)(
        dataset_params=args.dataset,
        tokenizer_path=args.tokenizer,
        context_len_char=args.context_len_char,
        **composers
    )
    preprocessor.prepare_model_input_parallel(dataset_path=prepared_dataset_path, num_workers=1)  # Don't change num_workers
    return prepared_dataset_path



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--model', '-m',
        type=str,
        choices=list(PREPROCESSORS.keys()),
        help='use preprocessor that is compatible with your model'
    )
    argparser.add_argument('--dataset', '-d', type=str, help='path to dataset')
    argparser.add_argument('--tokenizer', '-t', type=str, help='path to tokenizer')
    argparser.add_argument('--config_path', '-c', type=str, help='path to json config')
    argparser.add_argument(
        '--composers',
        type=str,
        choices=list(COMPOSERS.keys()),
        default='none',
        help='choose a context composer, if None uses the default composer from class'
    )
    argparser.add_argument('--out_dir', type=str, help='directory to save prepared dataset')

    args = argparser.parse_args()

    preprocess(args, {"lang_sep_symbol": "LANGSEP\n", "meta_info_sep_symbol": "METASEP\n",
                      "extension": ".py"})
