import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Dict

from llm_responder import build_llm_response, MODEL_REGISTRY
from api_llm_responder import build_api_llm_response


BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "Empty_Activity_android_studio_base_template"
OUTPUT_DIR = BASE_DIR / "output_projects"


@dataclass
class AgentProgress:
    notify: Callable[[str], None]


class AndroidAgent:
    def __init__(self, progress: Optional[AgentProgress] = None,
                 local_backend: Optional[str] = "gpt4all",
                 local_model: Optional[str] = "orca-mini-3b-gguf2-q4_0.gguf",
                 api_provider: Optional[str] = None,
                 api_model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.progress = progress or AgentProgress(lambda _: None)
        self.local_backend = (local_backend or "gpt4all").strip()
        self.local_model = (local_model or "orca-mini-3b-gguf2-q4_0.gguf").strip()
        self.api_provider = (api_provider or "").strip()
        self.api_model = (api_model or "").strip()
        self.api_key = (api_key or "").strip()
        self.should_stop = should_stop or (lambda: False)

    def _notify(self, message: str) -> None:
        try:
            self.progress.notify(message)
        except Exception:
            pass

    def _copy_template(self, project_name: str) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target_dir = OUTPUT_DIR / project_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(TEMPLATE_DIR, target_dir)
        # Remove machine-specific Gradle config if present
        local_props = target_dir / "local.properties"
        if local_props.exists():
            try:
                local_props.unlink()
            except Exception:
                pass
        # Stamp project name in settings.gradle.kts
        settings_gradle = target_dir / "settings.gradle.kts"
        if settings_gradle.exists():
            try:
                text = settings_gradle.read_text(encoding="utf-8")
                text = text.replace(
                    'rootProject.name = "Empty_Activity_android_studio_base_template"',
                    f'rootProject.name = "{project_name}"'
                )
                settings_gradle.write_text(text, encoding="utf-8")
            except Exception:
                pass
        return target_dir

    def _use_api(self) -> bool:
        return bool(self.api_provider and self.api_model and self.api_key)

    def _ask_app_name(self, idea: str) -> str:
        instruction = (
            "Choose a short, friendly Android app name for this idea. "
            "Respond ONLY with the name.\n\nIdea:" + idea
        )
        if self._use_api():
            name = build_api_llm_response(self.api_provider, self.api_model, self.api_key, instruction)
        else:
            model_spec = {
                "repo_id": "",
                "filename": self.local_model,
                "model_type": "llama",
                "backend": self.local_backend or "gpt4all",
            }
            name = build_llm_response(instruction, model_spec=model_spec)
        return (name or "MyApp").strip().replace("\n", " ")[:40]

    def _ask_architecture(self, idea: str, app_name: str) -> str:
        instruction = (
            "Design a simple, clean architecture for an Android app using Kotlin and XML. "
            "List the files to implement with brief purpose. Keep it minimal."
            f"\n\nApp: {app_name}\nIdea: {idea}"
        )
        if self._use_api():
            return build_api_llm_response(self.api_provider, self.api_model, self.api_key, instruction)
        model_spec = {
            "repo_id": "",
            "filename": self.local_model,
            "model_type": "llama",
            "backend": self.local_backend or "gpt4all",
        }
        return build_llm_response(instruction, model_spec=model_spec)

    def _llm_file_update(self, filepath: Path, friendly_label: str) -> None:
        try:
            existing = filepath.read_text(encoding="utf-8") if filepath.exists() else ""
        except Exception:
            existing = ""

        instruction = (
            "Given the existing file content, produce a JSON with filename and full replacement content. "
            "Return ONLY valid JSON: {\"filename\":..., \"content\":...}. "
            "Target a production-ready Android implementation that matches the app idea."
        )
        self._notify(friendly_label)
        if self._use_api():
            response = build_api_llm_response(self.api_provider, self.api_model, self.api_key, instruction, context=existing, progress_cb=self._notify)
        else:
            model_spec = {
                "repo_id": "",
                "filename": self.local_model,
                "model_type": "llama",
                "backend": self.local_backend or "gpt4all",
            }
            response = build_llm_response(instruction, context=existing, model_spec=model_spec)

        # Best-effort JSON extraction
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            data = json.loads(json_str)
            content = data.get("content", "")
            if not content:
                raise ValueError("Empty content from model")
        except Exception:
            # Fallback: write raw response
            content = response

        # Surface LLM response to UI (for user visibility)
        try:
            preview = response if len(response) <= 4000 else (response[:4000] + "\n...")
            self._notify(f"🧠 {filepath.name} response:\n{preview}")
        except Exception:
            pass

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")

    def run(self, idea: str) -> Path:
        # 1) Name selection
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        self._notify("⚙️ Setting up your app foundation...")
        app_name = self._ask_app_name(idea)

        # 2) Copy template
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        target_dir = self._copy_template(app_name)

        # 3) Plan
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        self._notify("🔍 Planning your app structure...")
        _ = self._ask_architecture(idea, app_name)

        # 4) File generation
        java_pkg_dir = target_dir / "app/src/main/java/com/example/empty_activity_android_studio_base_template"
        main_activity = java_pkg_dir / "MainActivity.kt"
        layout_main = target_dir / "app/src/main/res/layout/activity_main.xml"
        manifest = target_dir / "app/src/main/AndroidManifest.xml"
        app_gradle = target_dir / "app/build.gradle.kts"

        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        self._llm_file_update(main_activity, "📱 Creating your app's main screen...")
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        self._llm_file_update(layout_main, "🎨 Designing your app interface...")
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        self._llm_file_update(manifest, "🧭 Configuring your app settings...")
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        self._llm_file_update(app_gradle, "🧩 Finalizing your app build setup...")

        self._notify("✅ Your Android app is ready!")
        return target_dir

