import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QHBoxLayout, QFrame, QComboBox
)

from agent_tool import AndroidAgent, AgentProgress


APP_TITLE = "Android Agent Developer"


class WorkerThread(QThread):
    progress_signal = Signal(str)
    done_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, idea: str, model_key: str):
        super().__init__()
        self.idea = idea
        self.model_key = model_key

    def run(self):
        try:
            agent = AndroidAgent(progress=AgentProgress(self.progress_signal.emit), model_key=self.model_key)
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


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(760, 560)
        self._build_ui()
        self.worker: Optional[WorkerThread] = None

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Create your Android app from an idea")
        title.setObjectName("title")
        subtitle = QLabel("Describe your app in simple words. We'll handle the rest.")
        subtitle.setObjectName("subtitle")

        # Model selector
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "TinyLlama 1.1B Chat Q5",
            "Phi-2 Q4_K_M",
            "CodeLlama 7B Q4_K_M",
            "GPT4All orca-mini-3b Q4",
        ])
        self.model_combo.setCurrentText("TinyLlama 1.1B Chat Q5")

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
        layout.addWidget(QLabel("Model (for local use):"))
        layout.addWidget(self.model_combo)
        layout.addWidget(self.button)
        layout.addWidget(self.status)
        layout.addWidget(self.log)

        self.setStyleSheet(
            """
            QWidget { background: #0f1115; color: #E6E9EF; font-size: 14px; }
            #title { font-size: 24px; font-weight: 700; margin: 4px 0 0 0; }
            #subtitle { color: #A6ADBB; margin-bottom: 8px; }
            QTextEdit { background: #111318; border: 1px solid #2A2F3A; border-radius: 10px; padding: 8px; }
            QPushButton { background: #4CAF50; color: white; border: none; padding: 10px 16px; border-radius: 10px; font-weight: 700; }
            QPushButton:hover { background: #58C15E; }
            QPushButton:disabled { background: #2A2F3A; color: #6B7280; }
            #statusLine { background: #0C0E12; border: 1px solid #2A2F3A; border-radius: 10px; padding: 8px; }
            """
        )

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
        model_key = self.model_combo.currentText()
        self.worker = WorkerThread(idea, model_key)
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

