.PHONY: help install install-dev sync test lint format format-check check clean build
.PHONY: version release release-patch release-minor release-major publish-test publish info
.PHONY: dev worker db-upgrade db-downgrade db-migrate docker-build docker-run

# Extract current version from pyproject.toml
VERSION := $(shell grep -m1 'version = ' pyproject.toml | cut -d'"' -f2)

# Default target
.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(CYAN)ML API - Makefile Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  $(CYAN)make install$(NC)          Install production dependencies"
	@echo "  $(CYAN)make install-dev$(NC)      Install development dependencies"
	@echo "  $(CYAN)make sync$(NC)             Install/sync dependencies with uv"
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@echo "  $(CYAN)make lint$(NC)             Run linter (ruff)"
	@echo "  $(CYAN)make format$(NC)           Format code (black + ruff)"
	@echo "  $(CYAN)make format-check$(NC)     Check formatting without changes"
	@echo "  $(CYAN)make check$(NC)            Run all checks (format + lint)"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  $(CYAN)make test$(NC)             Run tests with coverage"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  $(CYAN)make dev$(NC)              Run development server"
	@echo "  $(CYAN)make worker$(NC)           Run background worker"
	@echo ""
	@echo "$(GREEN)Database:$(NC)"
	@echo "  $(CYAN)make db-upgrade$(NC)       Run database migrations"
	@echo "  $(CYAN)make db-downgrade$(NC)     Rollback last migration"
	@echo "  $(CYAN)make db-migrate$(NC)       Create new migration"
	@echo ""
	@echo "$(GREEN)Build:$(NC)"
	@echo "  $(CYAN)make build$(NC)            Build distribution packages"
	@echo "  $(CYAN)make clean$(NC)            Clean build artifacts"
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@echo "  $(CYAN)make docker-build$(NC)     Build Docker image"
	@echo "  $(CYAN)make docker-run$(NC)       Run Docker container"
	@echo ""
	@echo "$(GREEN)Release:$(NC)"
	@echo "  $(CYAN)make version$(NC)          Show current version"
	@echo "  $(CYAN)make release V=x.y.z$(NC)  Release version x.y.z (runs checks, tags, pushes)"
	@echo "  $(CYAN)make release-patch$(NC)    Bump patch version (0.1.0 -> 0.1.1)"
	@echo "  $(CYAN)make release-minor$(NC)    Bump minor version (0.1.0 -> 0.2.0)"
	@echo "  $(CYAN)make release-major$(NC)    Bump major version (0.1.0 -> 1.0.0)"
	@echo "  $(CYAN)make publish-test$(NC)     Publish to TestPyPI"
	@echo "  $(CYAN)make publish$(NC)          Publish to PyPI"
	@echo ""
	@echo "$(GREEN)Info:$(NC)"
	@echo "  $(CYAN)make info$(NC)             Show project information"

# =============================================================================
# Setup
# =============================================================================

install: ## Install production dependencies
	@echo "$(CYAN)Installing production dependencies...$(NC)"
	uv sync --no-dev
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(CYAN)Installing development dependencies...$(NC)"
	uv sync --extra dev
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

sync: ## Install/sync dependencies with uv (recommended)
	@echo "$(CYAN)Installing dependencies with uv...$(NC)"
	uv sync --extra dev
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linting (ruff)
	@echo "$(CYAN)Running ruff...$(NC)"
	uv run --with ruff ruff check ml_api/ tests/
	@echo "$(GREEN)✓ Linting passed$(NC)"

format: ## Format code with black and ruff
	@echo "$(CYAN)Formatting with black...$(NC)"
	uv run --with black black ml_api/ tests/
	@echo "$(CYAN)Auto-fixing with ruff...$(NC)"
	uv run --with ruff ruff check --fix ml_api/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check formatting without changes
	@echo "$(CYAN)Checking code format...$(NC)"
	uv run --with black black --check ml_api/ tests/
	uv run --with ruff ruff format --check ml_api/ tests/
	@echo "$(GREEN)✓ Format check passed$(NC)"

check: ## Run all checks (format check + lint)
	@echo "$(CYAN)Checking code format...$(NC)"
	uv run --with black black --check ml_api/ tests/
	@echo "$(CYAN)Running ruff...$(NC)"
	uv run --with ruff ruff check ml_api/ tests/
	@echo "$(GREEN)✓ All checks passed$(NC)"

# =============================================================================
# Testing
# =============================================================================

test: ## Run tests with coverage
	@echo "$(CYAN)Running tests...$(NC)"
	uv run --extra dev pytest tests/ -v --cov=ml_api --cov-report=term-missing --cov-report=xml
	@echo "$(GREEN)✓ Tests passed$(NC)"

# =============================================================================
# Development
# =============================================================================

dev: ## Run development server
	@echo "$(CYAN)Starting development server...$(NC)"
	uv run ml-api serve --reload --port 8000

worker: ## Run background worker
	@echo "$(CYAN)Starting background worker...$(NC)"
	uv run --with arq arq ml_api.workers.worker.WorkerSettings

# =============================================================================
# Database
# =============================================================================

db-upgrade: ## Run database migrations
	@echo "$(CYAN)Running database migrations...$(NC)"
	uv run --with alembic alembic upgrade head
	@echo "$(GREEN)✓ Migrations complete$(NC)"

db-downgrade: ## Rollback last migration
	@echo "$(CYAN)Rolling back last migration...$(NC)"
	uv run --with alembic alembic downgrade -1
	@echo "$(GREEN)✓ Rollback complete$(NC)"

db-migrate: ## Create new migration
	@read -p "Enter migration message: " message; \
	uv run --with alembic alembic revision --autogenerate -m "$$message"

# =============================================================================
# Build
# =============================================================================

build: clean ## Build distribution packages
	@echo "$(CYAN)Building distribution packages...$(NC)"
	uv build
	@echo "$(GREEN)✓ Build complete$(NC)"
	@echo "$(YELLOW)Packages:$(NC)"
	@ls -lh dist/

clean: ## Clean build artifacts and cache
	@echo "$(CYAN)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	rm -rf ml_api/__pycache__ cli/__pycache__ tests/__pycache__ **/__pycache__
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleaned$(NC)"

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker image
	@echo "$(CYAN)Building Docker image...$(NC)"
	docker build -t ml-api:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(CYAN)Running Docker container...$(NC)"
	docker run -p 8000:8000 --env-file .env ml-api:latest

# =============================================================================
# Release
# =============================================================================

# Show current version
version: ## Show current version
	@echo "Current version: $(VERSION)"

# Release a specific version: make release V=0.2.0
release: _check-version _pre-release ## Release specific version
	@echo "$(CYAN)Releasing version $(V)...$(NC)"
	@# Update version using uv
	uv version $(V)
	@# Commit, tag, and push
	git add pyproject.toml ml_api/__init__.py
	git commit -m "Release v$(V)"
	git tag -a "v$(V)" -m "Release v$(V)"
	git push origin main
	git push origin "v$(V)"
	@echo ""
	@echo "$(GREEN)✓ Released v$(V)$(NC)"
	@echo "  GitHub Actions will build and publish to PyPI"
	@echo "  Track progress: https://github.com/$$(git remote get-url origin | sed 's/.*github.com[:/]//;s/.git$$//')/actions"

# Bump patch version (0.1.0 -> 0.1.1)
release-patch: _pre-release ## Bump patch version
	@echo "$(CYAN)Bumping patch version: $(VERSION)$(NC)"
	@uv version --bump patch
	@NEW_VERSION=$$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2); \
	echo "$(CYAN)New version: $$NEW_VERSION$(NC)"; \
	git add pyproject.toml ml_api/__init__.py; \
	git commit -m "Release v$$NEW_VERSION"; \
	git tag -a "v$$NEW_VERSION" -m "Release v$$NEW_VERSION"; \
	git push origin main; \
	git push origin "v$$NEW_VERSION"; \
	echo ""; \
	echo "$(GREEN)✓ Released v$$NEW_VERSION$(NC)"; \
	echo "  GitHub Actions will build and publish to PyPI"; \
	echo "  Track progress: https://github.com/$$(git remote get-url origin | sed 's/.*github.com[:/]//;s/.git$$//')/actions"

# Bump minor version (0.1.0 -> 0.2.0)
release-minor: _pre-release ## Bump minor version
	@echo "$(CYAN)Bumping minor version: $(VERSION)$(NC)"
	@uv version --bump minor
	@NEW_VERSION=$$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2); \
	echo "$(CYAN)New version: $$NEW_VERSION$(NC)"; \
	git add pyproject.toml ml_api/__init__.py; \
	git commit -m "Release v$$NEW_VERSION"; \
	git tag -a "v$$NEW_VERSION" -m "Release v$$NEW_VERSION"; \
	git push origin main; \
	git push origin "v$$NEW_VERSION"; \
	echo ""; \
	echo "$(GREEN)✓ Released v$$NEW_VERSION$(NC)"; \
	echo "  GitHub Actions will build and publish to PyPI"; \
	echo "  Track progress: https://github.com/$$(git remote get-url origin | sed 's/.*github.com[:/]//;s/.git$$//')/actions"

# Bump major version (0.1.0 -> 1.0.0)
release-major: _pre-release ## Bump major version
	@echo "$(CYAN)Bumping major version: $(VERSION)$(NC)"
	@uv version --bump major
	@NEW_VERSION=$$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2); \
	echo "$(CYAN)New version: $$NEW_VERSION$(NC)"; \
	git add pyproject.toml ml_api/__init__.py; \
	git commit -m "Release v$$NEW_VERSION"; \
	git tag -a "v$$NEW_VERSION" -m "Release v$$NEW_VERSION"; \
	git push origin main; \
	git push origin "v$$NEW_VERSION"; \
	echo ""; \
	echo "$(GREEN)✓ Released v$$NEW_VERSION$(NC)"; \
	echo "  GitHub Actions will build and publish to PyPI"; \
	echo "  Track progress: https://github.com/$$(git remote get-url origin | sed 's/.*github.com[:/]//;s/.git$$//')/actions"

# Pre-release checks
_pre-release:
ifndef SKIP_CHECK
	@echo "$(CYAN)Running pre-release checks...$(NC)"
	@$(MAKE) format-check
	@$(MAKE) lint
	@$(MAKE) test
	@echo ""
	@echo "$(GREEN)✓ All checks passed$(NC)"
	@echo ""
endif

# Check that version is provided
_check-version:
ifndef V
	$(error Version not specified. Usage: make release V=x.y.z)
endif
	@if ! echo "$(V)" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$$'; then \
		echo "$(RED)Error: Invalid version format '$(V)'. Expected: x.y.z$(NC)"; \
		exit 1; \
	fi

# =============================================================================
# Publish (Manual - usually done by GitHub Actions)
# =============================================================================

publish-test: build ## Publish to TestPyPI
	@echo "$(CYAN)Publishing to TestPyPI...$(NC)"
	uv publish --publish-url https://test.pypi.org/legacy/
	@echo "$(GREEN)✓ Published to TestPyPI$(NC)"
	@echo "$(YELLOW)Test installation:$(NC)"
	@echo "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ml-api"

publish: build ## Publish to PyPI (manual - normally done via GitHub Actions)
	@echo "$(RED)Warning: This will publish to PyPI!$(NC)"
	@echo "Normally publishing is done automatically via GitHub Actions when you push a tag."
	@echo ""
	@read -p "Are you sure you want to manually publish? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo ""
	@echo "$(CYAN)Publishing to PyPI...$(NC)"
	uv publish
	@echo "$(GREEN)✓ Published to PyPI$(NC)"

# =============================================================================
# Info
# =============================================================================

info: ## Show project information
	@echo "$(CYAN)Project Information$(NC)"
	@echo "$(YELLOW)Name:$(NC)         ml-api"
	@echo "$(YELLOW)Version:$(NC)      $(VERSION)"
	@echo "$(YELLOW)Python:$(NC)       $$(python --version 2>&1)"
	@echo "$(YELLOW)UV:$(NC)           $$(uv --version 2>&1)"
	@echo ""
	@echo "$(CYAN)Dependencies$(NC)"
	@uv pip list | grep -E "(fastapi|polars|sqlalchemy|catboost|xgboost)" || echo "Run 'make sync' to install dependencies"
	@echo ""
	@echo "$(CYAN)Git Status$(NC)"
	@git status --short || echo "Not a git repository"
	@echo ""
	@echo "$(CYAN)Available Commands$(NC)"
	@echo "Run 'make help' for all available commands"
