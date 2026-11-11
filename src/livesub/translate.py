"""Translation through any OpenAI-compatible chat endpoint.

Offline by default in spirit: point it at a model on your own machine (Ollama, LM Studio, llama.cpp's
server, vLLM) and nothing leaves it. A hosted endpoint such as DeepSeek works the same way, if you
choose one. Nothing is configured unless you set LIVESUB_LLM_BASE_URL (or pass --llm-url).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

LANGUAGES = {
    "zh": "Simplified Chinese", "zh-tw": "Traditional Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese", "ru": "Russian", "it": "Italian",
    "ar": "Arabic", "hi": "Hindi", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "tr": "Turkish",
}


class TranslationError(RuntimeError):
    pass


def clean(reply: str) -> str:
    """The translation alone: without a reasoning model's <think> block, a "Translation:" label, or
    quotes around it."""
    text = re.sub(r"<think>.*?</think>", "", reply, flags=re.S | re.I)
    text = re.sub(r"^\s*(translation|译文|翻译)\s*[:：]\s*", "", text.strip(), flags=re.I)
    text = text.strip()
    pairs = {'"': '"', "'": "'", "“": "”", "「": "」", "『": "』"}
    if len(text) >= 2 and text[0] in pairs and text[-1] == pairs[text[0]]:
        text = text[1:-1].strip()
    return text


class ChatTranslator:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 30.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.cache: dict[tuple[str, str], str] = {}

    @classmethod
    def from_env(cls) -> ChatTranslator | None:
        url = os.environ.get("LIVESUB_LLM_BASE_URL")
        if not url:
            return None
        return cls(url, os.environ.get("LIVESUB_LLM_MODEL", "qwen2.5:7b"), os.environ.get("LIVESUB_LLM_API_KEY"))

    def check(self) -> None:
        """Can the endpoint be reached at all? Only a TCP connection is opened: no caption, no key and
        no request is sent. A wrong --llm-url then fails before the model loads."""
        import socket

        parts = urllib.parse.urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"{self.url}: the translation endpoint must be an http(s) URL")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            socket.create_connection((parts.hostname, port), timeout=3).close()
        except OSError as e:
            raise RuntimeError(f"cannot reach the translation endpoint {parts.scheme}://{parts.hostname}:{port} ({e}); "
                               "is the model server running?") from e

    def _opener(self) -> urllib.request.OpenerDirector:
        """A model on this machine is reached directly: an http_proxy setting would otherwise send
        every caption to the proxy first, which is exactly what running locally is meant to avoid."""
        host = urllib.parse.urlsplit(self.url).hostname or ""
        if host in ("localhost", "::1") or host.startswith("127.") or host.endswith(".localhost"):
            return urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return urllib.request.build_opener()

    def translate(self, text: str, target: str, context: list[str] | tuple[str, ...] = ()) -> str:
        key = (text, target)
        if key in self.cache:
            return self.cache[key]
        language = LANGUAGES.get(target.lower(), target)
        system = (f"You translate live subtitles into {language}. Reply with the translation of the line after "
                  "'Line:' only, with no quotes, notes or explanations. Lines marked 'Earlier:' are context "
                  "from just before; do not translate them.")
        user = "".join(f"Earlier: {c}\n" for c in context) + f"Line: {text}"
        body = {"model": self.model, "temperature": 0, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers, method="POST")
        try:
            with self._opener().open(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            reply = data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, TypeError, ValueError) as e:
            raise TranslationError(f"translation request to {self.url} failed: {e}") from e
        out = clean(reply or "")
        if not out:
            raise TranslationError("the translator returned an empty reply")
        self.cache[key] = out
        return out
