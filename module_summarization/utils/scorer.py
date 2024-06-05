import os

import numpy as np
from openai import OpenAI
import tiktoken
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OPENAI_SYSTEM_PROMPT = 'You are a code quality assesing engine.'

class OptionsScoringModel:
    def __init__(self,
                 model_name: str,
                 device: torch.DeviceObjType | str) -> None:
        if model_name in {'gpt-3.5-turbo', 'gpt-4'}:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
            token = os.getenv('OPENAI_API_KEY')
            if token is None:
                raise RuntimeError('The env variable OPENAI_API_KEY must be set!')
            self.model = OpenAI(api_key=token)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = self.model.to(device)
        self.model_name = model_name
        self.device = device

    def score_options(self, query: str, options: list[str]) -> torch.Tensor:
        if isinstance(self.model, OpenAI):
            return self._score_options_gpt(query, options)
        else:
            return self._score_options_transformers(query, options)

    def _score_options_transformers(self,
                                    query: str,
                                    options: list[str]) -> torch.Tensor:
        token_ids = self.tokenizer(query, return_tensors='pt')['input_ids']
        with torch.inference_mode():
            outs = self.model(token_ids.to(self.device))
        opt_tokens = []
        for opt in options:
            opt_token = self.tokenizer(opt, return_tensors='pt', add_special_tokens=False)['input_ids'][0, 0]
            opt_tokens.append(opt_token)
        log_probs = outs.logits[0, -1, opt_tokens].cpu()
        return log_probs

    def _score_options_gpt(self, query: str, options: list[str]) -> torch.Tensor:
        logit_bias = dict()
        for opt in options:
            tok_ids = self.tokenizer.encode(opt)
            assert len(tok_ids) == 1, 'Only single token options are supported'
            logit_bias[tok_ids[0]] = 100
        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=1,
            temperature=0.3,
            n=1,
            logprobs=True,
            top_logprobs=20,
            logit_bias=logit_bias
        )

        logprobs = np.full(2, np.nan)
        choice = completion.choices[0]
        opt_to_idx = {t: n for n, t in enumerate(options)}
        min_lp = 0
        for logprob_item in choice.logprobs.content[0].top_logprobs:
            tok = logprob_item.token
            lp = logprob_item.logprob
            min_lp = min(min_lp, lp)
            if tok in opt_to_idx:
                logprobs[opt_to_idx[tok]] = lp
        logprobs[np.isnan(logprobs)] = min_lp - 2.3  # approximately 10 times less than the minimal one
        assert not np.isnan(logprobs).any()
        return torch.from_numpy(logprobs)
