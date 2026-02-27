# BI-IDE v8 Makefile
# ==================
# Development automation commands

.PHONY: help install dev test build deploy lint format clean

# Colors
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m

help: ## Show this help message
	@echo "$(BLUE)BI-IDE v8$(NC) - Development Commands"
	@echo "====================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

# Installation
install: ## Install all dependencies
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(BLUE)Installing UI dependencies...$(NC)"
	cd ui && npm install
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing dev dependencies...$(NC)"
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy
	cd ui && npm install
	@echo "$(GREEN)✓ Dev installation complete$(NC)"

# Development
dev: ## Run development server
	@echo "$(BLUE)Starting development server...$(NC)"
	python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

dev-ui: ## Run UI development server
	@echo "$(BLUE)Starting UI dev server...$(NC)"
	cd ui && npm run dev

dev-all: ## Run both backend and frontend (requires tmux or separate terminals)
	@echo "$(BLUE)Starting all services...$(NC)"
	@make dev & make dev-ui

# Testing
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	python -m pytest tests/ -v --tb=short

test-coverage: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	python -m pytest tests/ -v --cov=api --cov=core --cov=hierarchy --cov=erp --cov=ai --cov-report=html --cov-report=term

test-smoke: ## Run smoke tests
	@echo "$(BLUE)Running smoke tests...$(NC)"
	python -m scripts.smoke_test

test-quick: ## Run quick tests only
	@echo "$(BLUE)Running quick tests...$(NC)"
	python -m pytest tests/test_coverage.py -v

# Building
build: ## Build UI for production
	@echo "$(BLUE)Building UI...$(NC)"
	cd ui && npm run build
	@echo "$(GREEN)✓ Build complete$(NC)"

build-docker: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose build

# Deployment
deploy: ## Deploy to production (requires DOMAIN and EMAIL)
	@echo "$(YELLOW)Deploying to production...$(NC)"
	@if [ -z "$(DOMAIN)" ]; then echo "$(RED)Error: DOMAIN not set$(NC)"; exit 1; fi
	@if [ -z "$(EMAIL)" ]; then echo "$(RED)Error: EMAIL not set$(NC)"; exit 1; fi
	./scripts/deploy-production.sh $(DOMAIN) $(EMAIL)

deploy-docker: ## Deploy using Docker Compose
	@echo "$(BLUE)Deploying with Docker...$(NC)"
	docker-compose -f docker-compose.prod.yml up -d

# Code Quality
lint: ## Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	flake8 api core hierarchy erp ai --max-line-length=100 --exclude=__pycache__,.venv
	cd ui && npm run lint

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black api core hierarchy erp ai --line-length=100
	isort api core hierarchy erp ai --profile=black
	cd ui && npm run format

type-check: ## Run type checking
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy api core hierarchy --ignore-missing-imports

# Database
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running migrations...$(NC)"
	alembic upgrade head

db-reset: ## Reset database (WARNING: Destroys data!)
	@echo "$(RED)WARNING: This will destroy all data!$(NC)"
	@read -p "Are you sure? (yes/no): " confirm && [ $$confirm = yes ] || exit 1
	alembic downgrade base
	alembic upgrade head
	@echo "$(GREEN)✓ Database reset$(NC)"

# Maintenance
clean: ## Clean temporary files
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov .pytest_cache 2>/dev/null || true
	rm -rf ui/dist 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	pip install -U -r requirements.txt
	cd ui && npm update
	@echo "$(GREEN)✓ Update complete$(NC)"

# Monitoring
logs: ## View logs
	docker-compose logs -f

status: ## Check system status
	@echo "$(BLUE)System Status:$(NC)"
	@echo "  Python: $$(python --version)"
	@echo "  Node: $$(node --version)"
	@echo "  NPM: $$(npm --version)"
	@echo "  Docker: $$(docker --version)"
	@echo ""
	@echo "$(BLUE)Services:$(NC)"
	@docker-compose ps 2>/dev/null || echo "  Docker not running"

# Backup
backup: ## Create database backup
	@echo "$(BLUE)Creating backup...$(NC)"
	@mkdir -p backups
	docker exec bi-ide-db pg_dump -U bi_ide bi_ide > backups/backup-$$(date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✓ Backup created in backups/$(NC)"

# Security
security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	pip safety check 2>/dev/null || echo "$(YELLOW)safety not installed, skipping$(NC)"
	npm audit --prefix ui 2>/dev/null || echo "$(YELLOW)npm audit failed$(NC)"

# Documentation
docs: ## Generate documentation
	@echo "$(BLUE)Generating docs...$(NC)"
	@echo "API documentation available at: http://localhost:8000/docs"

# Default
.DEFAULT_GOAL := help
