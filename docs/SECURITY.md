# Security

Scope:

- Basic hardening for a self-hosted TTS service.
- No governance, auditing, or compliance features.

## API key (optional but recommended)

- Support a single static API key via environment variable:
  - API_KEY=...
- Require header:
  - Authorization: Bearer <key>
  - or X-API-Key: <key>

## CORS

If UI is hosted separately:

- Set CORS_ORIGINS explicitly
- Do not use "\*" in production unless you understand the risk

## Rate limiting (recommended)

If exposed publicly:

- Apply rate limiting at reverse proxy (Nginx/Caddy) or in app middleware.
- Concurrency already limits GPU-heavy work; rate limiting helps protect request queues.

## File uploads

Voice clone reference uploads:

- Validate file type (wav/flac)
- Enforce size limits
- Store uploads under outputs/ or a temp directory
- Avoid executing or importing anything from user-provided content

## Network exposure

- Prefer running behind a reverse proxy with TLS.
- Do not expose internal filesystem paths in API responses; return run_id and controlled download routes instead.
