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
        self.app_name: str = ""
        self.idea_text: str = ""
        self.architecture_plan: str = ""
        self.generated_main_activity: str = ""
        self.generated_layout: str = ""
        self.generated_manifest: str = ""
        self.generated_gradle: str = ""
        self.package_name: str = "com.example.empty_activity_android_studio_base_template"

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

    def _llm_call(self, instruction: str, existing: str, extra_context: str) -> str:
        prompt = instruction
        if extra_context:
            prompt += "\n\nContext:\n" + extra_context
        if existing:
            prompt += "\n\nExisting file content:\n" + existing
        if self._use_api():
            return build_api_llm_response(self.api_provider, self.api_model, self.api_key, prompt, context=None, progress_cb=self._notify)
        model_spec = {
            "repo_id": "",
            "filename": self.local_model,
            "model_type": "llama",
            "backend": self.local_backend or "gpt4all",
        }
        return build_llm_response(prompt, context=None, model_spec=model_spec, progress_cb=self._notify)

    def _write_from_response(self, filepath: Path, response: str) -> str:
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
            content = response
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        # Surface LLM response
        try:
            preview = response if len(response) <= 4000 else (response[:4000] + "\n...")
            self._notify(f"🧠 {filepath.name} response:\n{preview}")
        except Exception:
            pass
        return content

    def _generate_main_activity(self, target_dir: Path) -> None:
        java_pkg_dir = target_dir / "app/src/main/java/com/example/empty_activity_android_studio_base_template"
        main_activity = java_pkg_dir / "MainActivity.kt"
        try:
            existing = main_activity.read_text(encoding="utf-8")
        except Exception:
            existing = ""
        instruction = (
            f"Create the main activity logic for an Android app: '{self.idea_text}'.\n"
            f"Constraints:\n"
            f"- Keep package EXACTLY '{self.package_name}'.\n"
            f"- File path is 'app/src/main/java/com/example/empty_activity_android_studio_base_template/MainActivity.kt'.\n"
            f"- Use a single Activity named MainActivity.\n"
            f"- Do NOT add new classes/files (like ViewModels). Keep code self-contained.\n"
            f"- Prefer simple Compose UI or classic Views; be minimal and working.\n"
            f"Return ONLY JSON with keys 'filename' and 'content'."
        )
        extra = f"App name: {self.app_name}\nArchitecture plan (optional):\n{self.architecture_plan}"
        self._notify("📱 Creating your app's main screen...")
        response = self._llm_call(instruction, existing, extra)
        self.generated_main_activity = self._write_from_response(main_activity, response)

    def _generate_layout(self, target_dir: Path) -> None:
        layout_main = target_dir / "app/src/main/res/layout/activity_main.xml"
        try:
            existing = layout_main.read_text(encoding="utf-8")
        except Exception:
            existing = ""
        instruction = (
            "Create the XML layout 'app/src/main/res/layout/activity_main.xml' that matches this MainActivity.kt.\n"
            "Constraints:\n- Keep it minimal.\n- Ensure any view IDs referenced by MainActivity exist.\n"
            "Return ONLY JSON with 'filename' and 'content'."
        )
        extra = (
            "MainActivity.kt just generated:\n" + self.generated_main_activity
        )
        self._notify("🎨 Designing your app interface...")
        response = self._llm_call(instruction, existing, extra)
        self.generated_layout = self._write_from_response(layout_main, response)

    def _generate_manifest(self, target_dir: Path) -> None:
        manifest = target_dir / "app/src/main/AndroidManifest.xml"
        try:
            existing = manifest.read_text(encoding="utf-8")
        except Exception:
            existing = ""
        instruction = (
            "Create 'app/src/main/AndroidManifest.xml' for this app.\n"
            "Constraints:\n"
            f"- Use activity name '.MainActivity' and theme from template.\n- Do NOT add new activities.\n- Keep package implicit (no 'package' attr).\n"
            "Return ONLY JSON with 'filename' and 'content'."
        )
        extra = (
            "MainActivity.kt:\n" + self.generated_main_activity +
            "\n\nactivity_main.xml:\n" + self.generated_layout
        )
        self._notify("🧭 Configuring your app settings...")
        response = self._llm_call(instruction, existing, extra)
        self.generated_manifest = self._write_from_response(manifest, response)

    def _generate_gradle(self, target_dir: Path) -> None:
        app_gradle = target_dir / "app/build.gradle.kts"
        try:
            existing = app_gradle.read_text(encoding="utf-8")
        except Exception:
            existing = ""
        instruction = (
            "Produce 'app/build.gradle.kts' compatible with the existing template.\n"
            "Constraints:\n- Do NOT reference non-existing modules (no project(\":domain\")).\n- Use only AndroidX + Compose basics if needed.\n- Keep versions managed by the versions catalog (libs).\n"
            "Return ONLY JSON with 'filename' and 'content'."
        )
        extra = (
            "MainActivity.kt:\n" + self.generated_main_activity
        )
        self._notify("🧩 Finalizing your app build setup...")
        response = self._llm_call(instruction, existing, extra)
        self.generated_gradle = self._write_from_response(app_gradle, response)

    def _update_strings_xml(self, target_dir: Path) -> None:
        strings_path = target_dir / "app/src/main/res/values/strings.xml"
        try:
            text = strings_path.read_text(encoding="utf-8")
            import re
            pattern = r"(<string\s+name=\"app_name\">)(.*?)(</string>)"
            replacement = r"\1" + self.app_name + r"\3"
            new_text = re.sub(pattern, replacement, text)
            if new_text != text:
                strings_path.write_text(new_text, encoding="utf-8")
        except Exception:
            pass

    def run(self, idea: str) -> Path:
        self.idea_text = idea
        # 1) Name selection
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        self._notify("⚙️ Setting up your app foundation...")
        self.app_name = self._ask_app_name(idea)

        # 2) Copy template
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        target_dir = self._copy_template(self.app_name)

        # 3) Plan
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        self._notify("🔍 Planning your app structure...")
        self.architecture_plan = self._ask_architecture(idea, self.app_name)

        # 4) File generation with chained context
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        self._generate_main_activity(target_dir)
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        self._generate_layout(target_dir)
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        self._generate_manifest(target_dir)
        if self.should_stop():
            raise RuntimeError("Operation cancelled")
        self._generate_gradle(target_dir)
        # Update app name in strings
        self._update_strings_xml(target_dir)

        self._notify("✅ Your Android app is ready!")
        return target_dir