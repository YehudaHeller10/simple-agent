import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QHBoxLayout, QFrame, QComboBox, QCheckBox, QDialog, QFormLayout
)

from agent_tool import AndroidAgent, AgentProgress
from config_store import load_config, save_config


APP_TITLE = "Android Agent Developer"


class WorkerThread(QThread):
    progress_signal = Signal(str)
    done_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, idea: str, local_model: str, api_mode: bool, api_provider: str, api_model: str, api_key: str):
        super().__init__()
        self.idea = idea
        self.local_model = local_model
        self.api_mode = api_mode
        self.api_provider = api_provider
        self.api_model = api_model
        self.api_key = api_key

    def run(self):
        try:
            agent = AndroidAgent(
                progress=AgentProgress(self.progress_signal.emit),
                local_backend="gpt4all",
                local_model=self.local_model,
                api_provider=self.api_provider if self.api_mode else None,
                api_model=self.api_model if self.api_mode else None,
                api_key=self.api_key if self.api_mode else None,
            )
            target = agent.run(self.idea)
            self.done_signal.emit(str(target))
        except Exception as exc:
            self.error_signal.emit(str(exc))


class StatusLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("statusLine")
        layout = QHBoxLayout(self)
        self.icon = QLabel("🟡")
        self.text = QLabel("Waiting to start...")
        layout.addWidget(self.icon)
        layout.addWidget(self.text)
        layout.addStretch()

    def set_info(self, msg: str):
        self.icon.setText("🟡")
        self.text.setText(msg)

    def set_ok(self, msg: str):
        self.icon.setText("✅")
        self.text.setText(msg)

    def set_error(self, msg: str):
        self.icon.setText("❌")
        self.text.setText(msg)


class SettingsDialog(QDialog):
    def __init__(self, parent=None, cfg=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.cfg = cfg or {}
        form = QFormLayout(self)
        # Local (GPT4All) model
        self.local_model_combo = QComboBox()
        self.local_model_combo.setEditable(True)
        self.local_model_combo.setPlaceholderText("GPT4All model filename, e.g. orca-mini-3b-gguf2-q4_0.gguf")
        # API
        self.api_enabled = QCheckBox("Enable API")
        self.api_provider = QComboBox()
        self.api_provider.addItems(["OpenRouter", "Gemini"])
        self.api_model = QLineEdit()
        self.api_model.setPlaceholderText("e.g. openrouter/auto or gemini-1.5-pro")
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("API Key")

        form.addRow("Local model (GPT4All)", self.local_model_combo)
        form.addRow(self.api_enabled)
        form.addRow("API Provider", self.api_provider)
        form.addRow("API Model", self.api_model)
        form.addRow("API Key", self.api_key)

        btns = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.cancel_btn = QPushButton("Cancel")
        btns.addWidget(self.save_btn)
        btns.addWidget(self.cancel_btn)
        form.addRow(btns)

        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def set_models(self, models):
        self.local_model_combo.clear()
        self.local_model_combo.addItems(models)

    def load_from_cfg(self):
        self.local_model_combo.setEditText(self.cfg.get("local", {}).get("model", "orca-mini-3b-gguf2-q4_0.gguf"))
        api = self.cfg.get("api", {})
        self.api_enabled.setChecked(bool(api.get("enabled", False)))
        self.api_provider.setCurrentText(api.get("provider", "OpenRouter"))
        self.api_model.setText(api.get("model", "openrouter/auto"))
        self.api_key.setText(api.get("key", ""))

    def dump_to_cfg(self):
        self.cfg.setdefault("local", {})
        self.cfg["local"]["backend"] = "gpt4all"
        self.cfg["local"]["model"] = self.local_model_combo.currentText().strip()
        self.cfg.setdefault("api", {})
        self.cfg["api"]["enabled"] = self.api_enabled.isChecked()
        self.cfg["api"]["provider"] = self.api_provider.currentText()
        self.cfg["api"]["model"] = self.api_model.text().strip()
        self.cfg["api"]["key"] = self.api_key.text().strip()
        return self.cfg


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(880, 640)
        self.cfg = load_config()
        self._init_menu()
        self._build_ui()
        self.worker: Optional[WorkerThread] = None

    def _init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        settings_menu = menubar.addMenu("Settings")
        about_menu = menubar.addMenu("About")

        self.settings_action = QAction("Preferences...", self)
        settings_menu.addAction(self.settings_action)
        self.settings_action.triggered.connect(self.open_settings)

        self.exit_action = QAction("Exit", self)
        file_menu.addAction(self.exit_action)
        self.exit_action.triggered.connect(self.close)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("Create your Android app from an idea")
        title.setObjectName("title")
        subtitle = QLabel("Describe your app in simple words. We'll handle the rest.")
        subtitle.setObjectName("subtitle")

        # No inline model/API controls; moved to Settings dialog

        self.input = QTextEdit()
        self.input.setPlaceholderText("e.g., A shopping list app with categories and reminders")
        self.input.setFixedHeight(100)

        self.button = QPushButton("Create my app")
        self.button.clicked.connect(self.start_agent)

        self.status = StatusLine()

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Progress will appear here...")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.input)
        layout.addWidget(self.button)
        layout.addWidget(self.status)
        layout.addWidget(self.log)

        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0e1014; color: #E6E9EF; font-size: 14px; }
            QMenuBar { background: #0e1014; color: #E6E9EF; }
            QMenu { background: #141821; color: #E6E9EF; }
            QMenu::item:selected { background: #1b2230; }
            #title { font-size: 28px; font-weight: 800; margin: 8px 0 0 0; }
            #subtitle { color: #A6ADBB; margin-bottom: 16px; }
            QTextEdit { background: #111318; border: 1px solid #2A2F3A; border-radius: 12px; padding: 12px; }
            QPushButton { background: #3B82F6; color: white; border: none; padding: 12px 18px; border-radius: 12px; font-weight: 700; }
            QPushButton:hover { background: #4C8EF9; }
            QPushButton:disabled { background: #2A2F3A; color: #6B7280; }
            #statusLine { background: #0C0E12; border: 1px solid #2A2F3A; border-radius: 12px; padding: 10px; }
            """
        )

    def open_settings(self):
        dlg = SettingsDialog(self, cfg=self.cfg.copy())
        # Try to list some known GPT4All model names (can be edited by user)
        default_models = [
            "orca-mini-3b-gguf2-q4_0.gguf",
            "mistral-7b-instruct-v0.2.Q4_0.gguf",
            "ggml-gpt4all-j-v1.3-groovy.bin",
        ]
        dlg.set_models(default_models)
        dlg.load_from_cfg()
        if dlg.exec() == QDialog.Accepted:
            self.cfg = dlg.dump_to_cfg()
            save_config(self.cfg)

    def append_log(self, text: str):
        self.log.append(text)
        if "✅" in text or text.lower().startswith("ai engine is ready"):
            self.status.set_ok(text)
        elif text.lower().startswith("error") or "❌" in text:
            self.status.set_error(text)
        else:
            self.status.set_info(text)

    def start_agent(self):
        idea = self.input.toPlainText().strip()
        if not idea:
            self.append_log("Please write a short description of your app idea.")
            return
        self.button.setDisabled(True)
        self.log.clear()
        self.append_log("Starting...")
        local_model = self.cfg.get("local", {}).get("model", "orca-mini-3b-gguf2-q4_0.gguf")
        api_cfg = self.cfg.get("api", {})
        api_mode = bool(api_cfg.get("enabled", False))
        api_provider = api_cfg.get("provider", "")
        api_model = api_cfg.get("model", "")
        api_key = api_cfg.get("key", "")
        self.worker = WorkerThread(idea, local_model, api_mode, api_provider, api_model, api_key)
        self.worker.progress_signal.connect(self.append_log)
        self.worker.done_signal.connect(self.on_done)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_done(self, target: str):
        self.append_log(f"✅ Your Android app is ready!\nSaved to: {target}")
        self.button.setDisabled(False)

    def on_error(self, message: str):
        self.append_log(f"❌ Something went wrong: {message}")
        self.button.setDisabled(False)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

