"""Settings → Assistant: provider settings, the write-only API key and model lists.
No model is reached: the provider calls are faked."""

from __future__ import annotations

import httpx
import pytest

from hax_gui import assistant as assistant_bridge

needs_assistant = pytest.mark.skipif(not assistant_bridge.available(), reason="hax_assistant is not installed")
SECRET = "sk-test-0123456789abcdef"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HAX_ASSISTANT_CONFIG", str(tmp_path / "assistant_cfg" / "settings.json"))
    for var in ("HAX_ASSISTANT_BACKEND", "HAX_ASSISTANT_BASE_URL", "HAX_ASSISTANT_MODEL", "HAX_ASSISTANT_API_KEY",
                "HAX_ASSISTANT_EMBED_MODEL", "HAX_ASSISTANT_EMBED_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HAX_ASSISTANT_DOCS", "off")
    assistant_bridge.reset_session()


@needs_assistant
def test_config_saves_and_reports_sources(client, run, monkeypatch):
    r = run(client.get("/api/assistant/config"))
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["sources"]["model"] == "default" and c["key"]["can_send"] is True
    r = run(client.put("/api/assistant/config", json={
        "backend": "openai", "base_url": "https://llm.example.org/v1", "model": "big-model",
        "embed_base_url": "http://localhost:11434/v1", "embed_model": "embeddinggemma:latest"}))
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["values"]["model"] == "big-model" and c["sources"]["model"] == "saved"
    assert c["values"]["embed_base_url"] == "http://localhost:11434/v1"
    monkeypatch.setenv("HAX_ASSISTANT_MODEL", "env-model")
    c = run(client.get("/api/assistant/config")).json()
    assert c["values"]["model"] == "env-model" and c["sources"]["model"] == "env"
    assert c["env_vars"]["model"] == "HAX_ASSISTANT_MODEL"


@needs_assistant
def test_config_refuses_a_key_and_bad_values(client, run):
    assert run(client.put("/api/assistant/config", json={"api_key": SECRET})).status_code == 422
    r = run(client.put("/api/assistant/config", json={"base_url": "llm.example.org"}))
    assert r.status_code == 422 and "http(s)" in r.json()["detail"]
    assert run(client.put("/api/assistant/config", json={"backend": "anthropic"})).status_code == 422


@needs_assistant
def test_key_is_write_only(client, run):
    from hax_assistant import credentials
    r = run(client.put("/api/assistant/config/key", json={"base_url": "https://llm.example.org/v1", "key": SECRET}))
    assert r.status_code == 200, r.text
    assert SECRET not in r.text and r.json()["origin"] == "https://llm.example.org"
    c = run(client.get("/api/assistant/config"))
    assert SECRET not in c.text and c.json()["key"]["origins"] == ["https://llm.example.org"]
    assert credentials.get_key("https://llm.example.org/v1") == SECRET
    r = run(client.delete("/api/assistant/config/key", params={"base_url": "https://llm.example.org/v1"}))
    assert r.status_code == 200 and r.json()["removed"] and r.json()["key"]["origins"] == []


@needs_assistant
def test_key_only_over_a_safe_channel(client, run):
    from hax_gui.app import app
    proxied = {"X-Forwarded-For": "203.0.113.7"}
    body = {"base_url": "https://llm.example.org/v1", "key": SECRET}
    r = run(client.put("/api/assistant/config/key", json=body, headers=proxied))
    assert r.status_code == 403 and "hax-assistant key set" in r.json()["detail"]
    assert run(client.get("/api/assistant/config", headers=proxied)).json()["key"]["can_send"] is False

    async def over_https():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://127.0.0.1") as c:
            return await c.put("/api/assistant/config/key", json=body, headers=proxied)
    assert run(over_https()).status_code == 200
    r = run(client.put("/api/assistant/config/key", json={"base_url": "http://gpu.lan:8000/v1", "key": SECRET}))
    assert r.status_code == 422 and "https" in r.json()["detail"]


@needs_assistant
def test_models_use_each_endpoints_own_key(client, run, monkeypatch):
    from hax_assistant.llm import providers
    calls = []

    def fake(kind, url, api_key=None, **kw):
        calls.append((kind, url, api_key))
        if "broken" in url:
            raise providers.ProviderError("could not reach broken.example.org")
        return {"server": None, "chat": [{"id": "m1", "size": None, "caps": []}], "embed": [], "hidden": []}

    monkeypatch.setattr(providers, "list_models", fake)
    run(client.put("/api/assistant/config/key", json={"base_url": "https://llm.example.org/v1", "key": SECRET}))
    r = run(client.post("/api/assistant/models", json={
        "backend": "openai", "base_url": "https://llm.example.org/v1", "embed_base_url": "http://localhost:11434/v1"}))
    assert r.status_code == 200 and SECRET not in r.text
    assert calls == [("openai", "https://llm.example.org/v1", SECRET), ("ollama", "http://localhost:11434/v1", None)]
    r = run(client.post("/api/assistant/models", json={"backend": "openai", "base_url": "https://broken.example.org/v1"}))
    assert r.json()["chat"] == {"ok": False, "error": "could not reach broken.example.org"}
    assert run(client.post("/api/assistant/models", json={"backend": "x", "base_url": "https://a.org"})).status_code == 422
