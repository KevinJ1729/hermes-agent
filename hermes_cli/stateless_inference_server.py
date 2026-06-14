"""Authenticated, stateless Codex OAuth inference for trusted internal apps."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

MAX_INSTRUCTIONS_CHARS = 50_000
MAX_CONTEXT_CHARS = 500_000


class ExtractionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(pattern=r"^(todo|decision|followup|issue)$")
    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(max_length=4_000)
    assignee_text: str | None = Field(default=None, max_length=500)
    due_date: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    priority: str = Field(pattern=r"^(high|normal|low)$")
    source_line_message_ids: list[int] = Field(max_length=200)


class ExtractionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: int = Field(gt=0)
    note: str | None = Field(default=None, max_length=4_000)
    status_change: str | None = Field(
        default=None,
        pattern=r"^(in_progress|done)$",
    )


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    creates: list[ExtractionCreate] = Field(max_length=200)
    updates: list[ExtractionUpdate] = Field(max_length=200)


class LineExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: str = Field(min_length=1, max_length=MAX_INSTRUCTIONS_CHARS)
    context: str = Field(min_length=1, max_length=MAX_CONTEXT_CHARS)


class UsageResponse(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class LineExtractionResponse(BaseModel):
    result: ExtractionResult
    model: str
    usage: UsageResponse


@dataclass(frozen=True)
class InferenceSettings:
    api_key: str
    model: str
    timeout_seconds: float = 120.0
    max_output_tokens: int = 8_192
    max_concurrency: int = 2

    @classmethod
    def from_env(cls) -> "InferenceSettings":
        api_key = os.getenv("HERMES_INFERENCE_API_KEY", "").strip()
        if len(api_key) < 32:
            raise RuntimeError(
                "HERMES_INFERENCE_API_KEY must contain at least 32 characters"
            )
        model = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
        if not model:
            raise RuntimeError("HERMES_INFERENCE_MODEL is required")
        return cls(
            api_key=api_key,
            model=model,
            timeout_seconds=float(
                os.getenv("HERMES_INFERENCE_TIMEOUT_SECONDS", "120")
            ),
            max_output_tokens=int(
                os.getenv("HERMES_INFERENCE_MAX_OUTPUT_TOKENS", "8192")
            ),
            max_concurrency=max(
                1, int(os.getenv("HERMES_INFERENCE_MAX_CONCURRENCY", "2"))
            ),
        )


def _oauth_configured() -> bool:
    auth_path = Path(get_hermes_home()) / "auth.json"
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    providers = payload.get("providers")
    if not isinstance(providers, dict):
        return False
    state = providers.get("openai-codex")
    if not isinstance(state, dict):
        return False
    tokens = state.get("tokens")
    return (
        isinstance(tokens, dict)
        and isinstance(tokens.get("access_token"), str)
        and bool(tokens["access_token"].strip())
        and isinstance(tokens.get("refresh_token"), str)
        and bool(tokens["refresh_token"].strip())
    )


def _extract_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("Codex returned no assistant content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Codex returned empty assistant content")
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 : -3].strip()
    return text


def _usage(response: Any) -> UsageResponse:
    raw = getattr(response, "usage", None)

    def read(*names: str) -> int:
        for name in names:
            value = getattr(raw, name, None)
            if value is None and isinstance(raw, dict):
                value = raw.get(name)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0
        return 0

    input_tokens = read("prompt_tokens", "input_tokens")
    output_tokens = read("completion_tokens", "output_tokens")
    return UsageResponse(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=read("total_tokens") or input_tokens + output_tokens,
        cache_read_tokens=read("cache_read_input_tokens", "cache_read_tokens"),
        cache_write_tokens=read(
            "cache_creation_input_tokens", "cache_write_tokens"
        ),
    )


def _call_codex(request: LineExtractionRequest, settings: InferenceSettings) -> Any:
    # This path deliberately bypasses AIAgent, sessions, memory, skills, and tools.
    from agent.auxiliary_client import call_llm

    messages = [
        {
            "role": "system",
            "content": (
                request.instructions.strip()
                + "\n\nReturn one JSON object only. Do not use markdown fences."
            ),
        },
        {"role": "user", "content": request.context},
    ]
    return call_llm(
        task=None,
        provider="openai-codex",
        model=settings.model,
        messages=messages,
        temperature=0,
        max_tokens=settings.max_output_tokens,
        timeout=settings.timeout_seconds,
        extra_body={
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "line_extraction",
                    "schema": ExtractionResult.model_json_schema(),
                    "strict": False,
                },
            }
        },
    )


def create_app(
    settings: InferenceSettings | None = None,
    caller: Callable[[LineExtractionRequest, InferenceSettings], Any] | None = None,
) -> FastAPI:
    resolved = settings or InferenceSettings.from_env()
    inference_caller = caller or _call_codex
    semaphore = asyncio.Semaphore(resolved.max_concurrency)

    app = FastAPI(
        title="Hermes Stateless Inference",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "oauth_configured": _oauth_configured(),
            "model": resolved.model,
        }

    @app.post("/v1/line-extractions", response_model=LineExtractionResponse)
    async def line_extractions(
        request: LineExtractionRequest,
        authorization: str | None = Header(default=None),
    ) -> LineExtractionResponse:
        expected = f"Bearer {resolved.api_key}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            )
        if not _oauth_configured():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Codex OAuth is not configured",
            )

        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Inference capacity is busy",
            ) from exc

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(inference_caller, request, resolved),
                timeout=resolved.timeout_seconds + 5,
            )
            parsed = ExtractionResult.model_validate_json(_extract_text(response))
            model = getattr(response, "model", None)
            return LineExtractionResponse(
                result=parsed,
                model=model.strip()
                if isinstance(model, str) and model.strip()
                else resolved.model,
                usage=_usage(response),
            )
        except (ValueError, ValidationError) as exc:
            logger.warning("Codex returned invalid extraction output: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Invalid structured inference response",
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Inference timed out",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Codex inference failed: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Inference provider failed",
            ) from exc
        finally:
            semaphore.release()

    return app


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
