.PHONY: help install install-dev lint format type test run-api run-ui pylint pylint-report

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
	@echo "  pylint       Run Pylint (quick check)"
	@echo "  pylint-report Run Pylint and save colorized report to file"

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
	omnivoice run-api --config configs/default.yaml

run-ui:
	omnivoice run-ui --config configs/default.yaml

pylint:
	pylint src --rcfile .pylintrc

pylint-report:
	pylint src --rcfile .pylintrc > pylint_report.txt 2>&1
	@echo "Report saved to pylint_report.txt"
