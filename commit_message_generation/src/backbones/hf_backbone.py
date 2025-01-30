import logging
from typing import Dict, Optional

import torch
from transformers import (  # type: ignore[import-untyped]
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)

from configs.backbones_configs import HFGenerationConfig, HFModelConfig
from src.prompts import CMGPrompt

from .base_backbone import CMGBackbone


class HuggingFaceBackbone(CMGBackbone):
    MODEL_NAME_TO_PROMPT_FORMAT: Dict[str, str] = {
        "bigcode/octocoder": "octocoder",
        "HuggingFaceH4/starchat-beta": "starchat",
        "HuggingFaceH4/starchat-alpha": "starchat",
        "Salesforce/instructcodet5p-16b": "alpaca",
    }
    name: str = "huggingface"

    def __init__(
        self,
        model_name: str,
        is_encoder_decoder: bool,
        model_kwargs: HFModelConfig,
        generation: HFGenerationConfig,
        device: str,
        prompt: Optional[CMGPrompt],
        use_bettertransformer: bool,
        seed: int,
    ):
        set_seed(seed)

        self._is_encoder_decoder = is_encoder_decoder
        self._name_or_path = model_name

        if self._is_encoder_decoder:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._name_or_path, **model_kwargs)  # type: ignore[arg-type]
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self._name_or_path, **model_kwargs)  # type: ignore[arg-type]
        if use_bettertransformer:
            try:
                self._model = self._model.to_bettertransformer()
            except Exception:
                logging.warning(
                    "Couldn't convert the model to BetterTransformer, proceeding with default implementation."
                )
        self._tokenizer = AutoTokenizer.from_pretrained(self._name_or_path)
        self._tokenizer.use_default_system_prompt = False
        self._model.eval()
        self._device = device
        if not model_kwargs.load_in_4bit and not model_kwargs.load_in_8bit:
            self._model.to(self._device)

        if generation.max_length is None and generation.max_new_tokens is None:
            generation.max_length = self._tokenizer.model_max_length
            logging.warning(
                f"Neither `max_length` nor `max_new_tokens` are passed, setting `max_length` to `model_max_length` of corresponding tokenizer ({generation.max_length})"
            )
        self._generation_config = GenerationConfig(**generation, eos_token_id=self._tokenizer.eos_token_id)  # type: ignore
        self._prompt = prompt

    @torch.inference_mode()
    def generate_msg(self, preprocessed_commit: Dict[str, str], **kwargs) -> Dict[str, Optional[str]]:
        if self._prompt:
            preprocessed_commit_mods = self._prompt.hf(
                preprocessed_commit,
                prompt_format=self.MODEL_NAME_TO_PROMPT_FORMAT.get(self._name_or_path, None),
                tokenizer=self._tokenizer,
            )
        else:
            assert "mods" in preprocessed_commit, (
                "HFBackbone expects 'mods' key to be present after preprocessing if no prompt is given."
            )
            preprocessed_commit_mods = preprocessed_commit["mods"]
        encoding = self._tokenizer(preprocessed_commit_mods, truncation=True, padding=False, return_tensors="pt")
        predictions = self._model.generate(
            encoding.input_ids.to(self._device),
            generation_config=self._generation_config,
        )

        # trim context and leave only generated part (not necessary for seq2seq models, bc context is supposed to go to encoder)
        if not self._is_encoder_decoder:
            predictions = predictions[:, encoding.input_ids.shape[1] :]

        decoded_predictions = self._tokenizer.batch_decode(predictions, skip_special_tokens=True)[0]
        return {"prediction": decoded_predictions}
