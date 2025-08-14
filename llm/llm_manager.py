from typing import List
import provider
from llm_base import BaseLLM
from llm_register import LLMRegistry
from llm_config import LLMConfig, ConfigLoader

DEFAULT_CONFIG_PATH = "configs/llm_config.yaml"

class LLMManager:

    def __init__(self):
        pass

    def load(self) -> List[BaseLLM]:
        llm_configs: List[LLMConfig] = ConfigLoader().load(DEFAULT_CONFIG_PATH)
        instances: List[BaseLLM] = []

        for cfg in llm_configs:
            if not cfg.enabled:
                continue

            provider = cfg.provider

            llm_class = LLMRegistry.get_provider(provider)
            if not llm_class:
                print(f"[Error] No registered LLM provider found: {provider}")
                continue
            try:
                llm_instance = llm_class(cfg)
                instances.append(llm_instance)
                print(f"[Info] Successfully created LLM instance: {provider} ({cfg.model})")
            except Exception as e:
                print(f"[Error] Failed to create LLM instance for {provider}: {e}")

        self._llm_instances = instances
        return instances

    @property
    def llms(self) -> List[BaseLLM]:
        return self._llm_instances



if __name__ == "__main__":
    print(LLMManager().load())