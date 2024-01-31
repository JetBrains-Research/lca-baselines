from dataclasses import dataclass
from typing import Type

from model_hub.model_classes import ModelBuilderBase, HFModelBuilder, HFModelBuilder4bit


@dataclass
class ModelMetainfo:
    builder: Type[ModelBuilderBase]
    checkpoint: str


MODEL_REGISTRY = {
    'starcoderbase-1b': ModelMetainfo(builder=HFModelBuilder, checkpoint="bigcode/starcoderbase-1b"),
    'starcoderbase-3b': ModelMetainfo(builder=HFModelBuilder, checkpoint="bigcode/starcoderbase-3b"),
    'starcoderbase-7b': ModelMetainfo(builder=HFModelBuilder, checkpoint="bigcode/starcoderbase-7b"),
    'starcoderbase': ModelMetainfo(builder=HFModelBuilder, checkpoint="bigcode/starcoderbase"),

    'starcoderbase-1b-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="bigcode/starcoderbase-1b"),
    'starcoderbase-3b-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="bigcode/starcoderbase-3b"),
    'starcoderbase-7b-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="bigcode/starcoderbase-7b"),
    'starcoderbase-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="bigcode/starcoderbase"),

    'codellama-7b': ModelMetainfo(builder=HFModelBuilder, checkpoint="codellama/CodeLlama-7b-hf"),
    'codellama-13b': ModelMetainfo(builder=HFModelBuilder, checkpoint="codellama/CodeLlama-13b-hf"),
    'codellama-34b': ModelMetainfo(builder=HFModelBuilder, checkpoint="codellama/CodeLlama-34b-hf"),

    'codellama-7b-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="codellama/CodeLlama-7b-hf"),
    'codellama-13b-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="codellama/CodeLlama-13b-hf"),
    'codellama-34b-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="codellama/CodeLlama-34b-hf"),

    # 'starcoder': 'model_hub.starcoder',
    # 'codegen2': 'model_hub.codegen2',
    # 'codegen25': 'model_hub.codegen2_5',
    # 'starcoder1b': 'model_hub.starcoder1b',
    # 'starcoder3b': 'model_hub.starcoder3b',
    # 'starcoder7b': 'model_hub.starcoder7b',
    # 'codellama7b': 'model_hub.codeLLama7b',
    # 'codellama7b_4bit': 'model_hub.codeLLama7b_4bit',
    # 'h3_pretrained_fl': 'fl-safari.fl-pipeline.scripts.lca.load_safari_model'
}