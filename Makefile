# Makefile for DiscoSeqSampler development

.PHONY: help pre-commit setup-dev test

# Default target
help:
	@echo "Available targets:"
	@echo "  pre-commit   - Install and run pre-commit hooks"
	@echo "  setup-dev    - Setup development environment"
	@echo "  test         - Run tests"

# Pre-commit
pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg
	pre-commit run --all-files

# Development setup
setup-dev:
	pip install -e .[dev]
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "Development environment setup complete!"

# Test
test:
	python -m pytest tests/ -v
