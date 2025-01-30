import os
from typing import Any, Dict, Optional

from src.prompts import CMGPrompt

from .openai_backbone import OpenAIBackbone


class DeepSeekBackbone(OpenAIBackbone):
    name: str = "deepseek"

    def __init__(
        self,
        model_name: str,
        prompt: CMGPrompt,
        parameters: Dict[str, Any],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # https://api-docs.deepseek.com/
        super().__init__(
            model_name=model_name,
            prompt=prompt,
            parameters=parameters,
            api_key=api_key if api_key else os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
