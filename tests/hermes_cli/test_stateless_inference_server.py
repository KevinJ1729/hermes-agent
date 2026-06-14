from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from hermes_cli.stateless_inference_server import (
    InferenceSettings,
    create_app,
)


def _response(content: str):
    return SimpleNamespace(
        model="gpt-test",
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content)),
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
        ),
    )


def _client(monkeypatch, tmp_path: Path, caller):
    auth = {
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "access",
                    "refresh_token": "refresh",
                }
            }
        }
    }
    (tmp_path / "auth.json").write_text(json.dumps(auth), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    settings = InferenceSettings(
        api_key="x" * 32,
        model="gpt-test",
        timeout_seconds=1,
        max_concurrency=1,
    )
    return TestClient(create_app(settings=settings, caller=caller))


def test_rejects_missing_bearer_token(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path, lambda *_args: None)
    response = client.post(
        "/v1/line-extractions",
        json={"instructions": "Extract", "context": "Message"},
    )
    assert response.status_code == 401


def test_returns_validated_structured_output(monkeypatch, tmp_path):
    payload = {
        "creates": [
            {
                "type": "todo",
                "title": "Prepare report",
                "summary": "",
                "assignee_text": None,
                "due_date": None,
                "priority": "normal",
                "source_line_message_ids": [1],
            }
        ],
        "updates": [],
    }
    client = _client(
        monkeypatch,
        tmp_path,
        lambda *_args: _response(json.dumps(payload)),
    )
    response = client.post(
        "/v1/line-extractions",
        headers={"Authorization": f"Bearer {'x' * 32}"},
        json={"instructions": "Extract", "context": "Message"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["result"] == payload
    assert body["model"] == "gpt-test"
    assert body["usage"]["total_tokens"] == 14


def test_rejects_invalid_structured_output(monkeypatch, tmp_path):
    client = _client(
        monkeypatch,
        tmp_path,
        lambda *_args: _response('{"creates": [{"type": "invalid"}]}'),
    )
    response = client.post(
        "/v1/line-extractions",
        headers={"Authorization": f"Bearer {'x' * 32}"},
        json={"instructions": "Extract", "context": "Message"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "Invalid structured inference response"


def test_health_reports_missing_oauth(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    settings = InferenceSettings(api_key="x" * 32, model="gpt-test")
    client = TestClient(create_app(settings=settings, caller=lambda *_args: None))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "oauth_configured": False,
        "model": "gpt-test",
    }
