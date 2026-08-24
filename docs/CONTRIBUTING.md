# Contributing

Welcome to OmniVoice Studio! We value your contributions. Please follow these guidelines:

## 📐 Development Standard

- **Python Version**: Use Python 3.12+ (managed with `conda` or `venv`).
- **Clean Code**: Adhere to SOLID principles. Every new feature should be accompanied by unit tests.

## 🧹 Code Quality (Linting)

We use a multi-layered linting approach to ensure high performance and maintainability:

1. **Ruff**: Primary, high-speed linter and formatter.
2. **Pylint**: Deep static analysis and code duplication checks.
3. **MyPy**: Static type checking for type safety.
4. **Codespell**: Spell checking for code and documentation.

### Running Linting Locally

Before pushing your changes, ensure they pass all quality checks:

```bash
# High-speed check (Ruff)
make lint
make format

# Deep analysis (Pylint)
make pylint

# Comprehensive type checking (MyPy)
make type
```

### 📄 Exporting Lint Reports

If you need to share a detailed report (e.g., for code review or debugging), you can generate a text report:

```bash
# Generates pylint_report.txt in the root directory
make pylint-report
```

## 🧪 Testing

- **Quick Test**: Run `make test` (pytest).
- **Assets**: Keep generated audio artifacts out of version control (`outputs/` is ignored).

## 🚀 CI/CD Integration

Our GitHub Action and GitLab CI pipelines enforce these standards. Pull requests will not be merged unless:

- The build is green (Passing Ruff, Pylint ≥ 7.0, MyPy, and Pytest).
- All spell-check errors are resolved in `.codespell-ignore-words.txt`.

---

Thank you for contributing to OmniVoice Studio! 🎙️✨
