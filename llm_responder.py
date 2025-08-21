import os
from typing import Optional, Callable, Dict, Any

from huggingface_hub import hf_hub_download


DEFAULT_MODEL_REPO = os.environ.get("ANDROID_AGENT_MODEL_REPO", "TheBloke/CodeLlama-7B-GGUF")
DEFAULT_MODEL_FILE = os.environ.get("ANDROID_AGENT_MODEL_FILE", "codellama-7b.Q4_K_M.gguf")
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".android_agent", "models")

# Friendly presets for local models (small, CPU-friendly first)
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "TinyLlama 1.1B Chat Q5": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
        "model_type": "llama",
    },
    "Phi-2 Q4_K_M": {
        "repo_id": "TheBloke/phi-2-GGUF",
        "filename": "phi-2.Q4_K_M.gguf",
        "model_type": "phi",
    },
    "CodeLlama 7B Q4_K_M": {
        "repo_id": "TheBloke/CodeLlama-7B-GGUF",
        "filename": "codellama-7b.Q4_K_M.gguf",
        "model_type": "llama",
    },
}


class GGUFModelManager:
    """Lightweight manager for downloading and running a local GGUF LLM.

    Designed to load on CPU-only laptops using quantized models.
    """

    def __init__(self,
                 repo_id: str = DEFAULT_MODEL_REPO,
                 filename: str = DEFAULT_MODEL_FILE,
                 cache_dir: str = MODEL_CACHE_DIR,
                 model_type: str = "llama",
                 progress_cb: Optional[Callable[[str], None]] = None):
        self.repo_id = repo_id
        self.filename = filename
        self.cache_dir = cache_dir
        self.model_type = model_type
        self.progress_cb = progress_cb or (lambda msg: None)
        self._backend = None  # "llama_cpp" or "ctransformers"
        self._llm = None

        os.makedirs(self.cache_dir, exist_ok=True)

    def _notify(self, message: str) -> None:
        try:
            self.progress_cb(message)
        except Exception:
            pass

    def ensure_model(self) -> str:
        """Ensures the GGUF file exists locally, downloading if needed."""
        local_path = os.path.join(self.cache_dir, self.filename)
        if os.path.exists(local_path):
            return local_path

        self._notify("Downloading the AI model (first time only)...")
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            cache_dir=self.cache_dir,
            local_dir=self.cache_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        self._notify("Model download complete.")
        return local_path

    def load(self) -> None:
        if self._llm is not None:
            return
        model_path = self.ensure_model()
        self._notify("Starting the AI engine...")
        # Try llama.cpp first
        try:
            from llama_cpp import Llama  # type: ignore
            self._llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=max(2, os.cpu_count() or 4),
                n_gpu_layers=0,
                verbose=False
            )
            self._backend = "llama_cpp"
        except Exception:
            # Fallback to ctransformers (pure wheels available on Windows)
            from ctransformers import AutoModelForCausalLM  # type: ignore
            self._llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type=self.model_type or "llama",
                gpu_layers=0,
                context_length=4096
            )
            self._backend = "ctransformers"
        self._notify("AI engine is ready.")

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        if self._llm is None:
            self.load()
        assert self._llm is not None

        prompt = f"<s>[INST]<<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt}[/INST]"
        if self._backend == "llama_cpp":
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stop=["</s>"]
            )
            return output["choices"][0]["text"].strip()
        else:
            # ctransformers generation API
            return self._llm(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stop=["</s>"]
            )


def build_llm_response(task_instruction: str,
                       context: Optional[str] = None,
                       progress_cb: Optional[Callable[[str], None]] = None,
                       model_spec: Optional[Dict[str, str]] = None) -> str:
    """High-level helper that initializes the model and returns a response.

    This is the single entry the agent uses.
    """
    model_spec = model_spec or {
        "repo_id": DEFAULT_MODEL_REPO,
        "filename": DEFAULT_MODEL_FILE,
        "model_type": "llama",
    }
    manager = GGUFModelManager(
        repo_id=model_spec.get("repo_id", DEFAULT_MODEL_REPO),
        filename=model_spec.get("filename", DEFAULT_MODEL_FILE),
        model_type=model_spec.get("model_type", "llama"),
        progress_cb=progress_cb,
    )

    system_prompt = (
        "You are an expert Android app developer. Always respond in valid JSON when asked, "
        "using {\"filename\":..., \"content\":...}. Keep code concise, compilable, and production quality."
    )
    user_prompt = task_instruction
    if context:
        user_prompt += "\n\nExisting content:\n" + context

    return manager.generate(system_prompt, user_prompt)

