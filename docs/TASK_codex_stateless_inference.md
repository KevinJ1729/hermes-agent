# TASK_codex_stateless_inference

## Owner

Codex

## Branch

`codex/stateless-inference`

## Goal

Provide a dedicated, authenticated inference service for GH Dashboard that
uses Hermes-managed OpenAI Codex OAuth without loading Hermes conversations,
memory, workspace context, skills, or tools.

## Scope

- A standalone FastAPI server module and console entry point.
- A fixed LINE extraction response schema.
- Bearer-token authentication, request-size limits, timeouts, and bounded
  concurrency.
- Focused unit tests and deployment documentation.

The service may read only Hermes OAuth state from `HERMES_HOME/auth.json`.
It must not instantiate `AIAgent`, `SessionDB`, gateway sessions, memory
providers, plugins, or tool registries.

## Acceptance

- `/health` reports service and OAuth readiness without exposing secrets.
- `/v1/line-extractions` rejects missing or invalid bearer tokens.
- Requests are limited in size and produce only the fixed extraction shape.
- Model calls use `provider=openai-codex`, contain no tools, and do not persist
  prompts or responses in Hermes state.
- Tests cover authentication, validation, successful structured output, and
  upstream failures.
