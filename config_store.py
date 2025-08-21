import json
import os
from typing import Any, Dict


CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")


DEFAULT_CONFIG: Dict[str, Any] = {
    "mode": "local",  # 'local' or 'api'
    "local": {
        "backend": "gpt4all",
        "model": "orca-mini-3b-gguf2-q4_0.gguf",
    },
    "api": {
        "provider": "OpenRouter",
        "model": "openrouter/auto",
        "key": "",
    },
    "chats": [],  # list of {id, title, idea, project_path, created_at}
}


def load_config() -> Dict[str, Any]:
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {**DEFAULT_CONFIG, **data}
    except Exception:
        pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: Dict[str, Any]) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

