from llm.llm_base import BaseModel
from llm.llm_config import ModelConfig
from llm.llm_register import ModelRegistry
from openai import OpenAI
from typing import Tuple, Optional
from utils.status import RequestStatus


@ModelRegistry.register(provider="openai")
class OpenAILLM(BaseModel):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = OpenAI(api_key=self.api_key)

    def request(self, system_prompt: str,
                prompt: str) -> Tuple[bool, Optional[str]]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return RequestStatus.SUCCESS, response.choices[0].message.content
        except Exception as e:
            return RequestStatus.FAILURE, f"Error: {str(e)}"
