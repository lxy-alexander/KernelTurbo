from abc import ABC, abstractmethod
from llm_config import LLMConfig

class BaseLLM(ABC):

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.provider
        self.model = config.model
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.top_p = config.top_p

    @abstractmethod
    def request(self, system_prompt: str, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def info(self) -> str:
        return (
            f"Provider: {self.provider}\n"
            f"Model: {self.model}\n"
            f"API Key: {len(self.api_key)}\n"
            f"Base URL: {self.base_url}\n"
            f"Max Tokens: {self.max_tokens}\n"
            f"Temperature: {self.temperature}\n"
            f"Top P: {self.top_p}"
        )
