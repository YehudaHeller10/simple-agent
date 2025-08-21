import os
import json
import time
from typing import Optional, Callable
import requests


def _notify(cb: Optional[Callable[[str], None]], msg: str) -> None:
    try:
        if cb:
            cb(msg)
    except Exception:
        pass


def _build_system_prompt() -> str:
    return (
        "You are an expert Android app developer. When asked to modify a file, "
        "respond ONLY in raw JSON with keys 'filename' and 'content'. No prose."
    )


def build_api_llm_response(
    provider: str,
    model: str,
    api_key: str,
    task_instruction: str,
    context: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """Unified API LLM call for OpenRouter and Gemini.

    Returns the raw text response. Caller is responsible for JSON extraction.
    """
    provider = (provider or "").lower()
    if provider not in ("openrouter", "gemini"):
        raise ValueError("Unsupported provider. Choose 'OpenRouter' or 'Gemini'.")

    system_prompt = _build_system_prompt()
    user_prompt = task_instruction
    if context:
        user_prompt += "\n\nExisting content:\n" + context

    if provider == "openrouter":
        _notify(progress_cb, "Contacting OpenRouter...")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional ranking headers per OpenRouter docs
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_X_TITLE", "Android Agent Developer"),
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "top_p": 0.9,
        }
        # Simple retry with backoff for transient errors
        transient = {408, 429, 500, 502, 503, 504, 524, 529}
        last_err = None
        for attempt in range(4):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code in transient:
                    raise requests.HTTPError(f"{resp.status_code} {resp.reason}")
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_err = e
                if attempt < 3:
                    wait = 2 ** attempt
                    _notify(progress_cb, f"OpenRouter busy (retrying in {wait}s)...")
                    time.sleep(wait)
                else:
                    break
        raise RuntimeError("OpenRouter request failed. Please try again or choose another model.") from last_err

    # Gemini
    _notify(progress_cb, "Contacting Gemini...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"}
                ],
            }
        ],
        "generationConfig": {"temperature": 0.2, "topP": 0.9, "maxOutputTokens": 1024},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()

