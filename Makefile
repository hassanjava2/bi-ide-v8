# BI-IDE v8 Makefile
# ==================
# أتمتة عمليات التطوير | Development automation

.PHONY: help install install-dev dev dev-ui dev-all test test-coverage \
        test-smoke test-quick build build-docker deploy deploy-docker \
        lint format type-check db-migrate db-reset db-backup db-restore \
        clean update logs status backup security-check docs \
        docker-up docker-down docker-logs docker-build docker-push \
        migrate rollback migration-create seed services-start services-stop \
        health-check format-check ci-check ci-run

# Colors | الألوان
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m

# Variables | المتغيرات
PYTHON := python3
PIP := pip3
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := bi-ide

# ===========================================
# Help | المساعدة
# ===========================================

help: ## عرض رسالة المساعدة | Show this help message
	@echo "$(BLUE)BI-IDE v8$(NC) - أوامر التطوير | Development Commands"
	@echo "================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ===========================================
# Installation | التثبيت
# ===========================================

install: ## تثبيت جميع المتطلبات | Install all dependencies
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(BLUE)Installing UI dependencies...$(NC)"
	cd ui && npm install
	@echo "$(GREEN)✓ Installation complete | اكتمل التثبيت$(NC)"

install-dev: ## تثبيت متطلبات التطوير | Install development dependencies
	@echo "$(BLUE)Installing dev dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-cov black isort flake8 mypy pre-commit
	cd ui && npm install
	@echo "$(GREEN)✓ Dev installation complete | اكتمل تثبيت التطوير$(NC)"

install-prod: ## تثبيت متطلبات الإنتاج | Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements-prod.txt
	@echo "$(GREEN)✓ Production installation complete$(NC)"

# ===========================================
# Development | التطوير
# ===========================================

dev: ## تشغيل خادم التطوير | Run development server
	@echo "$(BLUE)Starting development server...$(NC)"
	$(PYTHON) -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

dev-ui: ## تشغيل خادم واجهة المستخدم | Run UI development server
	@echo "$(BLUE)Starting UI dev server...$(NC)"
	cd ui && npm run dev

dev-all: ## تشغيل الخادم وواجهة المستخدم | Run both backend and frontend
	@echo "$(BLUE)Starting all services...$(NC)"
	make dev & make dev-ui

# ===========================================
# Testing | الاختبارات
# ===========================================

test: ## تشغيل جميع الاختبارات | Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --tb=short

test-coverage: ## تشغيل الاختبارات مع التغطية | Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest tests/ -v \
		--cov=api --cov=core --cov=hierarchy --cov=erp --cov=ai \
		--cov-report=html --cov-report=term

test-smoke: ## تشغيل اختبارات الدخان | Run smoke tests
	@echo "$(BLUE)Running smoke tests...$(NC)"
	$(PYTHON) scripts/smoke_test.py

test-quick: ## تشغيل الاختبارات السريعة فقط | Run quick tests only
	@echo "$(BLUE)Running quick tests...$(NC)"
	$(PYTHON) -m pytest tests/test_coverage.py -v

test-specific: ## تشغيل اختبار محدد (استخدم: make test-specific TEST=test_name)
	@echo "$(BLUE)Running specific test: $(TEST)...$(NC)"
	$(PYTHON) -m pytest tests/$(TEST) -v

# ===========================================
# Building | البناء
# ===========================================

build: ## بناء واجهة المستخدم للإنتاج | Build UI for production
	@echo "$(BLUE)Building UI...$(NC)"
	cd ui && npm run build
	@echo "$(GREEN)✓ Build complete | اكتمل البناء$(NC)"

build-docker: ## بناء صور Docker | Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build

build-docker-gpu: ## بناء صور Docker مع GPU | Build Docker images with GPU
	@echo "$(BLUE)Building GPU Docker images...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.gpu.yml build

build-all: build build-docker ## بناء كل شيء | Build everything

# ===========================================
# Docker | دوكر
# ===========================================

docker-up: ## تشغيل خدمات Docker | Start Docker services
	@echo "$(BLUE)Starting Docker services...$(NC)"
	$(DOCKER_COMPOSE) up -d

docker-down: ## إيقاف خدمات Docker | Stop Docker services
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	$(DOCKER_COMPOSE) down

docker-logs: ## عرض سجلات Docker | View Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-ps: ## عرض حالة الحاويات | Show container status
	$(DOCKER_COMPOSE) ps

docker-build: ## بناء وإعادة تشغيل | Build and restart
	@echo "$(BLUE)Rebuilding and restarting...$(NC)"
	$(DOCKER_COMPOSE) up -d --build

docker-push: ## دفع الصور إلى السجل | Push images to registry
	@echo "$(BLUE)Pushing Docker images...$(NC)"
	docker push $(REGISTRY)/bi-ide-api:latest
	docker push $(REGISTRY)/bi-ide-worker:latest

docker-clean: ## تنظيف Docker | Clean up Docker
	@echo "$(YELLOW)Cleaning Docker...$(NC)"
	docker system prune -f
	docker volume prune -f

# ===========================================
# Deployment | النشر
# ===========================================

deploy: ## نشر للإنتاج | Deploy to production
	@echo "$(YELLOW)Deploying to production...$(NC)"
	@if [ -z "$(DOMAIN)" ]; then echo "$(RED)Error: DOMAIN not set$(NC)"; exit 1; fi
	@if [ -z "$(EMAIL)" ]; then echo "$(RED)Error: EMAIL not set$(NC)"; exit 1; fi
	./scripts/deploy-production.sh $(DOMAIN) $(EMAIL)

deploy-docker: ## نشر باستخدام Docker | Deploy using Docker Compose
	@echo "$(BLUE)Deploying with Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.prod.yml up -d

deploy-vps: ## نشر على VPS | Deploy to VPS
	@echo "$(BLUE)Deploying to VPS...$(NC)"
	./scripts/deploy-vps.sh

deploy-check: ## التحقق من حالة النشر | Check deployment status
	@echo "$(BLUE)Checking deployment status...$(NC)"
	curl -s http://localhost:8000/health | jq .

# ===========================================
# Code Quality | جودة الكود
# ===========================================

lint: ## تشغيل الفاحص | Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	flake8 api core hierarchy erp ai --max-line-length=100 --exclude=__pycache__,.venv,venv
	cd ui && npm run lint

format: ## تنسيق الكود | Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black api core hierarchy erp ai --line-length=100
	isort api core hierarchy erp ai --profile=black
	cd ui && npm run format

format-check: ## التحقق من التنسيق | Check formatting
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check api core hierarchy erp ai --line-length=100
	isort --check-only api core hierarchy erp ai --profile=black

type-check: ## فحص الأنواع | Run type checking
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy api core hierarchy --ignore-missing-imports

security-check: ## فحص الأمان | Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	-safety check 2>/dev/null || echo "$(YELLOW)safety not installed$(NC)"
	-npm audit --prefix ui 2>/dev/null || echo "$(YELLOW)npm audit failed$(NC)"

# ===========================================
# CI/CD | التكامل المستمر
# ===========================================

ci-check: format-check lint type-check test-quick ## جميع فحوصات CI | All CI checks
	@echo "$(GREEN)✓ All CI checks passed$(NC)"

ci-run: ## تشغيل خط أنابيب CI | Run CI pipeline
	@echo "$(BLUE)Running CI pipeline...$(NC)"
	make ci-check
	make test-coverage

# ===========================================
# Database | قاعدة البيانات
# ===========================================

db-migrate: ## تشغيل ترحيلات قاعدة البيانات | Run database migrations
	@echo "$(BLUE)Running migrations...$(NC)"
	alembic upgrade head

db-reset: ## إعادة تعيين قاعدة البيانات (⚠️ يحذف البيانات!) | Reset database
	@echo "$(RED)WARNING: This will destroy all data! | تحذير: سيحذف جميع البيانات!$(NC)"
	@read -p "Are you sure? (yes/no): " confirm && [ $$confirm = yes ] || exit 1
	alembic downgrade base
	alembic upgrade head
	@echo "$(GREEN)✓ Database reset | تمت إعادة التعيين$(NC)"

db-backup: ## نسخ احتياطي لقاعدة البيانات | Create database backup
	@echo "$(BLUE)Creating backup...$(NC)"
	@mkdir -p backups
	docker exec bi-ide-db pg_dump -U bi_ide bi_ide > backups/backup-$$(date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✓ Backup created in backups/$(NC)"

db-restore: ## استعادة نسخة احتياطية | Restore database backup
	@echo "$(BLUE)Restoring from backup...$(NC)"
	@if [ -z "$(BACKUP_FILE)" ]; then echo "$(RED)Error: BACKUP_FILE not set$(NC)"; exit 1; fi
	docker exec -i bi-ide-db psql -U bi_ide bi_ide < $(BACKUP_FILE)
	@echo "$(GREEN)✓ Restore complete$(NC)"

db-seed: ## إضافة بيانات تجريبية | Seed database with test data
	@echo "$(BLUE)Seeding database...$(NC)"
	$(PYTHON) scripts/seed_data.py

migrate: db-migrate ## اختصار | Shortcut

rollback: ## التراجع عن آخر ترحيل | Rollback last migration
	@echo "$(BLUE)Rolling back last migration...$(NC)"
	alembic downgrade -1

migration-create: ## إنشاء ترحيل جديد | Create new migration
	@if [ -z "$(MESSAGE)" ]; then echo "$(RED)Error: MESSAGE not set$(NC)"; exit 1; fi
	alembic revision --autogenerate -m "$(MESSAGE)"

# ===========================================
# Services | الخدمات
# ===========================================

services-start: ## تشغيل جميع الخدمات | Start all services
	@echo "$(BLUE)Starting all services...$(NC)"
	$(PYTHON) scripts/start_services.py

services-stop: ## إيقاف جميع الخدمات | Stop all services
	@echo "$(BLUE)Stopping all services...$(NC)"
	-pkill -f "uvicorn api.app:app"
	-pkill -f "core.tasks"

services-restart: services-stop services-start ## إعادة تشغيل الخدمات | Restart services

health-check: ## فحص صحة النظام | Health check
	@echo "$(BLUE)Running health checks...$(NC)"
	$(PYTHON) scripts/health_check.py

# ===========================================
# Maintenance | الصيانة
# ===========================================

clean: ## تنظيف الملفات المؤقتة | Clean temporary files
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf ui/dist 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean docker-clean ## تنظيف شامل | Deep clean
	rm -rf .venv venv node_modules ui/node_modules

update: ## تحديث المتطلبات | Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install -U -r requirements.txt
	cd ui && npm update
	@echo "$(GREEN)✓ Update complete$(NC)"

logs: ## عرض السجلات | View logs
	tail -f logs/bi-ide.log 2>/dev/null || $(DOCKER_COMPOSE) logs -f

status: ## التحقق من حالة النظام | Check system status
	@echo "$(BLUE)System Status | حالة النظام:$(NC)"
	@echo "  Python: $$($(PYTHON) --version)"
	@echo "  Node: $$(node --version 2>/dev/null || echo 'not installed')"
	@echo "  NPM: $$(npm --version 2>/dev/null || echo 'not installed')"
	@echo "  Docker: $$(docker --version 2>/dev/null || echo 'not installed')"
	@echo ""
	@echo "$(BLUE)Services | الخدمات:$(NC)"
	-$(DOCKER_COMPOSE) ps 2>/dev/null || echo "  Docker not running | دوكر لا يعمل"

# ===========================================
# Backup | النسخ الاحتياطي
# ===========================================

backup: db-backup ## نسخ احتياطي | Create backup
	@echo "$(BLUE)Creating full backup...$(NC)"
	tar -czf backups/full-backup-$$(date +%Y%m%d_%H%M%S).tar.gz \
		uploads/ logs/ backups/*.sql 2>/dev/null || true
	@echo "$(GREEN)✓ Full backup created$(NC)"

# ===========================================
# Documentation | التوثيق
# ===========================================

docs: ## عرض التوثيق | Generate/view documentation
	@echo "$(BLUE)Documentation | التوثيق:$(NC)"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo "  ReDoc: http://localhost:8000/redoc"

# ===========================================
# Default | الافتراضي
# ===========================================

.DEFAULT_GOAL := help
