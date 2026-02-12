# tests/

Pytest suite for the project.

Notes:

- Some tests are “standalone” (they validate repo files like voices/profiles JSON).
- Tests that depend on `src/` are written to skip automatically until those modules exist.

Run:

- pytest -q
