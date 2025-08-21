## Android Agent Developer

One-click tool to generate complete Android Studio projects from a plain English idea. Runs a local GGUF LLM to design and write code, wrapped in a friendly consumer UI.

### Quick Start (development)

1. Create a Python 3.10+ virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python main.py
   ```

The first run downloads a compact coding model (GGUF). Subsequent runs use the cached model.

### Packaging (PyInstaller)

To build a standalone executable:
```bash
pyinstaller android_agent.spec --noconfirm
```
Artifacts are in `dist/AndroidAgent/`. On Windows, run the EXE. On Linux/macOS, run the binary in that folder.

### Template

The app uses the Android Studio Empty Activity template from `Empty_Activity_android_studio_base_template`. The original is never modified; each run copies to `output_projects/{ProjectName}`.

### Notes

- Offline use is supported after the first model download.
- For best results, keep ideas concise and specific (e.g., "shopping list with categories and share").
