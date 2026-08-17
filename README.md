# llm-gateway

This is a public, cleaned-up release of a personal tool I built during my independent engineering period in 2025. It is a small gateway that sits between your applications and LLM providers. Your code calls one OpenAI-compatible endpoint. The gateway picks a provider, handles failures, tracks spend, and logs every request as structured JSON.

## What it does

- One endpoint, `POST /v1/chat/completions`, in the standard OpenAI request format.
- Named routes map model aliases such as `fast` or `quality` to chains of provider and model pairs.
- Fallback across the chain when a provider fails or times out, with per-route retry rules.
- Two routing strategies: `priority` (in listed order) and `round_robin` (rotate across targets).
- Three providers: OpenAI, Anthropic, and a built-in `mock` provider that runs fully offline.
- Cost tracking against a static price table, daily budget limits per API key, and a usage report endpoint. Spend is stored in SQLite.
- Structured JSON logs per request: route, provider, latency, tokens, cost, fallback hops.
- An optional evals hook that scores each response after it is served and logs the score.

The Anthropic adapter translates requests and responses between the OpenAI format and the Anthropic Messages API in both directions.

Streaming is not supported in this release. Requests with `stream: true` get a 400.

## Quickstart

Runs offline with no API keys. The example config routes everything to the mock provider.

```bash
git clone <this repo> && cd llm-gateway
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
GATEWAY_CONFIG=examples/routes.yaml uvicorn llm_gateway.app:create_app --factory --port 8080
```

Then in another terminal:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-key" \
  -d '{
    "model": "fast",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

You get a normal chat completion response from the mock provider. Check spend:

```bash
curl -s "http://localhost:8080/v1/usage?key=demo-key"
```

Or with Docker:

```bash
docker compose up --build
```

To use real providers, export `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and uncomment the matching targets in `examples/routes.yaml`.

## Routing config reference

Config is one YAML file, validated at startup. Path comes from the `GATEWAY_CONFIG` environment variable.

```yaml
routes:
  fast:                      # clients send model: "fast"
    strategy: priority       # or round_robin
    retry:
      attempts: 2            # tries per target before moving on
      timeout_seconds: 15    # per attempt
      backoff_seconds: 0.2   # sleep between attempts
    targets:                 # tried in order (priority) or rotated (round_robin)
      - provider: openai
        model: gpt-4o-mini
      - provider: anthropic
        model: claude-haiku-4-5

budgets:
  default_daily_usd: 5.0     # daily USD limit for any key without its own entry
  keys:
    demo-key: 1.0            # per-key overrides

pricing:                     # optional overrides for the built-in price table
  gpt-4o-mini:
    input_per_1k: 0.00015
    output_per_1k: 0.0006

evals:
  enabled: true
  evaluators:
    - empty_or_refusal

database_path: usage.db      # SQLite file for usage accounting
```

`GET /v1/routes` returns the loaded routes so you can confirm what is live.

## Fallback behaviour

The router works through a route's targets in order. Each target gets up to `retry.attempts` tries, each bounded by `timeout_seconds`.

- Timeouts and retryable errors (HTTP 429 and 5xx from the provider) consume attempts, then the router moves to the next target.
- Non-retryable errors (provider 4xx, missing API key) skip the remaining attempts on that target and move on at once.
- With `round_robin`, the starting target rotates per request, and failures still fall through the rest of the chain.
- When every target fails, the gateway returns 502 with the per-target errors in the message.

The number of targets tried beyond the first is logged as `fallback_hops` on every request.

## Cost policies

A static price table in `pricing.py` holds USD prices per 1K input and output tokens. These are illustrative prices, update for your providers. Config entries under `pricing` override the built-ins. Unknown models cost zero rather than blocking traffic.

Each request's cost is computed from the token usage the provider reports and added to the caller's daily total in SQLite. The API key is taken from the `Authorization: Bearer` header; requests without one are accounted under `anonymous`.

When a key's spend for the day reaches its limit, further requests get 429 with code `daily_budget_exceeded` until the next UTC day. A key set to `0.0` is blocked outright. `GET /v1/usage` reports requests, tokens, and spend per key per day, and `?key=` filters to one key.

## Observability

Every request produces one JSON log line on stdout:

```json
{"ts": 1755500000.0, "level": "info", "event": "request_completed",
 "request_id": "9b0f6f0f4a2c4d1e", "api_key": "demo-key", "route": "fast",
 "provider": "mock", "model": "mock-small", "latency_ms": 1.8,
 "prompt_tokens": 2, "completion_tokens": 9, "cost_usd": 0.0,
 "attempts": 1, "fallback_hops": 0}
```

Failed requests log `request_failed` with the per-target errors. Budget refusals log `budget_exceeded`.

- `GET /healthz` returns `{"status": "ok"}` for liveness checks.
- `GET /v1/routes` returns the live routing table.
- `GET /v1/usage` returns spend per key per day.

## Evals hook

After each successful response the gateway can run evaluators: small offline scorers that see the request and response and return a score. Scores are logged as `eval_scored` events, they never change or fail the request.

One evaluator ships with the gateway. `empty_or_refusal` scores 0.0 for an empty response, 0.2 when the response opens with a refusal phrase, and 1.0 otherwise. It is a cheap signal for spotting routes that return nothing useful.

To add your own, implement the `Evaluator` protocol in `evals.py`, register it in `EVALUATOR_REGISTRY`, and list its name in the config. Evaluators must not make external calls.

## Tests

```bash
pytest
```

The whole suite runs offline against the mock provider.

## Status

Status: v0.1.0, single-maintainer, reviewed releases. Streaming, per-request price lookup from provider APIs, and hot config reload are not implemented.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
