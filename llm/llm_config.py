import os
import yaml
import re
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from pydantic import Field

# Load environment variables from .env file
load_dotenv()


@dataclass
class ModelConfig:
    provider: str = Field(..., description="Provider name, required")
    api_key: str = Field(..., description="API key in .env, required")
    model: str = Field(..., description="Model name, required")
    base_url: str = Field(..., description="Base URL, required")
    enabled: bool = False
    max_tokens: int = 4096
    temperature: int = 0
    top_p: float = 1.0


class ConfigLoader:
    ENV_PATTERN = re.compile(r"\$\{(\w+)\}")  # Pattern for ${VAR_NAME}

    def __init__(self):
        pass

    def _replace_env_var(self, value: str) -> str:
        """Replace ${VAR} with environment variable value."""
        if isinstance(value, str):
            match = self.ENV_PATTERN.fullmatch(value)
            if match:
                var_name = match.group(1)
                env_value = os.environ.get(var_name)
                if env_value is None:
                    raise RuntimeError(
                        f"[Error] Missing required environment variable: {var_name}"
                    )
                return env_value
        return value

    def _recursive_replace(self, obj):
        """Recursively traverse dict/list and replace environment placeholders."""
        if isinstance(obj, dict):
            return {k: self._recursive_replace(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._recursive_replace(item) for item in obj]
        else:
            return self._replace_env_var(obj)

    def load(self, config_yaml_path: str) -> List[ModelConfig]:
        if not config_yaml_path:
            raise ValueError("[Error] YAML config path cannot be None.")

        try:
            with open(config_yaml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)

            replaced = self._recursive_replace(raw)

            if not isinstance(replaced, list):
                raise ValueError(
                    "LLM config should be a list of provider dicts.")

            # Only load enabled configs
            llm_configs = [
                ModelConfig(**cfg) for cfg in replaced
                if cfg.get("enabled", True)
            ]
            return llm_configs
        except Exception as e:
            print(f"[Error] Failed to load config: {e}")
            raise


if __name__ == "__main__":
    print(ConfigLoader().load())
