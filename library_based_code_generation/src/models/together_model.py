import os

from together import Together

from .example_generation_model import ExampleGenerationModel


class TogetherModel(ExampleGenerationModel):

    def __init__(self, model_name: str, use_bm25: bool = False):
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.model_name = model_name
        self.use_bm25 = use_bm25

    def generate(self, task_description: str, project_apis: list[str] = None) -> str:
        instruction = self.get_prompt(task_description) \
            if not self.use_bm25 \
            else self.get_bm25_prompt(task_description, project_apis)

        prompt = [
            {
                "role": "user",
                "content": instruction
            },
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=0.0,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content

    def name(self):
        if not self.use_bm25:
            return self.model_name
        else:
            return f"bm25/{self.model_name}"
