from typing import Dict, Type, List
from llm_base import BaseLLM

class LLMRegistry:
    _providers: Dict[str, Type[BaseLLM]] = {}
    
    @classmethod
    def register(cls, provider: str):
        def decorator(llm_cls: Type[BaseLLM]):
            cls._providers[provider] = llm_cls
            # print(f"[Info] Registered provider: {provider}")
            return llm_cls
        return decorator
    
    @classmethod
    def get_provider(cls, name: str) -> Type[BaseLLM]:
        if name not in cls._providers:
            raise ValueError(f"[Error] Unknown provider: {name}. Available: {cls.list_providers()}")
        return cls._providers[name]
    
    @classmethod
    def list_providers(cls) -> List[str]:
        return list(cls._providers.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._providers