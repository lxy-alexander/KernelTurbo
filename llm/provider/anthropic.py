from llm.llm_base import BaseModel
from llm.llm_config import ModelConfig
from llm.llm_register import ModelRegistry
import anthropic
from typing import Tuple, Optional
from utils.status import RequestStatus


@ModelRegistry.register(provider="anthropic")
class AnthropicLLM(BaseModel):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def request(self, system_prompt: str,
                prompt: str) -> Tuple[bool, Optional[str]]:
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            print(response)
            if response is None:
                raise ValueError("Empty response from API")
            return RequestStatus.SUCCESS, response.content[0].text
        except Exception as e:
            return RequestStatus.FAILURE, f"Error: {str(e)}"
