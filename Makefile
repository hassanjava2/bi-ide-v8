# BI IDE v8 - Makefile
# Commands for development and deployment

.PHONY: help install dev test lint format docker-build docker-up docker-down clean

help:
	@echo "BI IDE v8 - Available Commands:"
	@echo ""
	@echo "  make install      - Install Python dependencies"
	@echo "  make dev          - Run in development mode"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make lint         - Run linting (ruff, black, mypy)"
	@echo "  make format       - Format code with black"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up    - Start with Docker Compose"
	@echo "  make docker-down  - Stop Docker Compose"
	@echo "  make clean        - Clean temporary files"
	@echo ""

install:
	pip install -r requirements.txt

dev:
	set PYTHONIOENCODING=utf-8 && python start.py

test:
	pytest -v --cov=core --cov=hierarchy --cov-report=html

test-unit:
	pytest -v -m unit

lint:
	ruff check .
	black --check .
	mypy core/ hierarchy/ --ignore-missing-imports

format:
	black .

docker-build:
	docker build -t bi-ide-api:latest -f Dockerfile .
	docker build -t bi-ide-ui:latest -f ui/Dockerfile ./ui
	docker build -t bi-ide-worker:latest -f Dockerfile.worker .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage

# Production deployment commands
deploy-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

backup:
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/ learning_data/

restore:
	@echo "Usage: make restore FILE=backup_YYYYMMDD_HHMMSS.tar.gz"
	tar -xzf $(FILE)
