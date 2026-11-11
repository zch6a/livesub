import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from livesub.translate import ChatTranslator, TranslationError, clean


class FakeLLM:
    """A local OpenAI-compatible endpoint that replies with a canned answer and records requests."""

    def __init__(self, reply="你好。", status=200):
        self.reply, self.status, self.requests = reply, status, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                data = json.dumps({"choices": [{"message": {"content": fake.reply}}]}).encode()
                self.send_response(fake.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def llm():
    f = FakeLLM()
    yield f
    f.close()


def test_sends_the_line_with_context_and_returns_the_translation(llm):
    t = ChatTranslator(llm.url, "local-model", api_key="k")
    assert t.translate("Hello.", "zh", ["Earlier line."]) == "你好。"
    req = llm.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer k"
    assert req["body"]["model"] == "local-model" and req["body"]["temperature"] == 0
    assert "Simplified Chinese" in req["body"]["messages"][0]["content"]
    assert req["body"]["messages"][1]["content"] == "Earlier: Earlier line.\nLine: Hello."


def test_repeated_lines_come_from_the_cache(llm):
    t = ChatTranslator(llm.url, "m")
    t.translate("Hi.", "zh")
    t.translate("Hi.", "zh")
    t.translate("Hi.", "ja")
    assert len(llm.requests) == 2


@pytest.mark.parametrize("reply, expected", [
    ("<think>The user wants Chinese.</think>\n你好。", "你好。"),
    ("Translation: Bonjour.", "Bonjour."),
    ("“你好。”", "你好。"),
    ('"Hola."', "Hola."),
    ("译文：你好", "你好"),
])
def test_cleans_what_local_models_add(reply, expected):
    assert clean(reply) == expected


def test_failures_raise_a_translation_error():
    bad = FakeLLM(status=500)
    try:
        with pytest.raises(TranslationError):
            ChatTranslator(bad.url, "m").translate("x", "zh")
    finally:
        bad.close()
    empty = FakeLLM(reply="<think>hmm</think>")
    try:
        with pytest.raises(TranslationError, match="empty"):
            ChatTranslator(empty.url, "m").translate("x", "zh")
    finally:
        empty.close()
    with pytest.raises(TranslationError):
        ChatTranslator("http://127.0.0.1:9/v1", "m", timeout=2).translate("x", "zh")


def test_not_configured_means_no_translator(monkeypatch):
    monkeypatch.delenv("LIVESUB_LLM_BASE_URL", raising=False)
    assert ChatTranslator.from_env() is None
    monkeypatch.setenv("LIVESUB_LLM_BASE_URL", "http://localhost:11434/v1")
    t = ChatTranslator.from_env()
    assert t is not None and t.url == "http://localhost:11434/v1/chat/completions"


def test_a_local_model_is_reached_directly_even_with_a_proxy_configured(llm, monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")  # nothing listens there
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    assert ChatTranslator(llm.url, "m").translate("Hello.", "zh") == "你好。"
