.PHONY: help install install-dev lint format type test run-api run-ui

help:
	@echo "Targets:"
	@echo "  install      Install runtime deps (use requirements.txt)"
	@echo "  install-dev  Install editable + dev extras"
	@echo "  lint         Ruff lint"
	@echo "  format       Ruff format"
	@echo "  type         MyPy"
	@echo "  test         Pytest"
	@echo "  run-api      Run FastAPI server"
	@echo "  run-ui       Run Gradio UI"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

type:
	mypy src

test:
	pytest -q

run-api:
	tts-platform run-api --config configs/default.yaml

run-ui:
	tts-platform run-ui --config configs/default.yaml
