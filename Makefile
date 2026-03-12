.PHONY: install install-dev test lint docker-build clean help

PYTHON ?= python
PIP    ?= pip

help:
	@echo "Available targets:"
	@echo "  install       Install the package and core dependencies"
	@echo "  install-dev   Install with dev extras (pytest, ruff)"
	@echo "  test          Run the test suite"
	@echo "  lint          Run ruff linter"
	@echo "  docker-build  Build the Docker image"
	@echo "  clean         Remove build artifacts and __pycache__ dirs"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	$(PYTHON) -m ruff check finger_impedance/ scripts/ examples/

docker-build:
	docker build -t finger-impedance-analysis .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
