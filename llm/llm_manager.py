from typing import List
from llm.llm_base import BaseModel
from llm.llm_register import ModelRegistry
from llm.llm_config import ModelConfig, ConfigLoader

DEFAULT_CONFIG_PATH = "configs/llm_config.yaml"


class ModelManager:

    def __init__(self):
        pass

    def load_models_from_config(self, config_path: str = None) -> List[BaseModel]:
        llm_configs: List[ModelConfig] = ConfigLoader().load(
            config_path or DEFAULT_CONFIG_PATH)
        instances: List[BaseModel] = []

        for cfg in llm_configs:
            if not cfg.enabled:
                continue
            provider = cfg.provider
            llm_class = ModelRegistry.get_provider(provider)
            if not llm_class:
                print(f"[Error] No registered LLM provider found: {provider}")
                continue
            try:
                llm_instance = llm_class(cfg)
                instances.append(llm_instance)
                print(
                    f"[Info] Successfully created LLM instance: {provider} ({cfg.model})"
                )
            except Exception as e:
                print(
                    f"[Error] Failed to create LLM instance for {provider}: {e}"
                )
        self._llm_instances = instances
        return instances

    @property
    def llms(self) -> List[BaseModel]:
        return self._llm_instances


if __name__ == "__main__":
    print(ModelManager().load())
