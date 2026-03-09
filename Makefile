# =============================================================================
# CONFIGURATION
# =============================================================================

# Project configuration
PROJECT_NAME = dl-nlp
PROJECT_PACKAGE = dl-nlp
PYTHON = python3
UV = uv

# Test configuration

# Colors for output
BLUE = \033[34m
GREEN = \033[32m
YELLOW = \033[33m
RED = \033[31m
NC = \033[0m # No Color

.DEFAULT_GOAL := help

# =============================================================================
# HELP SYSTEM
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)AutoDS Research Project Commands$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@echo ""
	@echo "$(GREEN)Installation & Setup:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { if ($$1 ~ /^(install|uv-|venv)/) printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { if ($$1 ~ /^test/) printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { if ($$1 ~ /^(lint|format|pre-commit|mypy|quality)/) printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { if ($$1 ~ /^(clean|env-check|status|reset)/) printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Services:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { if ($$1 ~ /^(searx-)/) printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Development Shortcuts:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { if ($$1 ~ /^(dev|quick|full)/) printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# =============================================================================
# INSTALLATION & SETUP
# =============================================================================

.PHONY: install
install: ## Install project dependencies locally (recommended for development)
	@echo "$(BLUE)Installing dependencies with uv...$(NC)"
	${UV} sync --all-extras
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

.PHONY: install-prod
install-prod: ## Install production dependencies only
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	${UV} sync --no-dev
	@echo "$(GREEN)Production dependencies installed!$(NC)"

.PHONY: update
update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	${UV} sync --upgrade
	@echo "$(GREEN)Dependencies updated!$(NC)"

.PHONY: uv-venv
uv-venv: ## Create virtual environment using uv
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	${UV} venv
	@echo "$(GREEN)Virtual environment created!$(NC)"

.PHONY: uv-sync
uv-sync: ## Install all dependencies using uv
	@echo "$(BLUE)Syncing dependencies...$(NC)"
	${UV} sync --all-extras
	@echo "$(GREEN)Dependencies synced!$(NC)"

.PHONY: install-dev
install-dev: uv-venv uv-sync ## Create venv and install all dependencies (recommended for development)
	@echo "$(GREEN)Development environment ready!$(NC)"

# =============================================================================
# TESTING
# =============================================================================

.PHONY: test
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	${UV} run pytest
	@echo "$(GREEN)Tests completed!$(NC)"

.PHONY: test-verbose
test-verbose: ## Run tests with verbose output
	@echo "$(BLUE)Running tests with verbose output...$(NC)"
	${UV} run pytest -v --tb=short --continue-on-collection-errors
	@echo "$(GREEN)Verbose tests completed!$(NC)"

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	${UV} run pytest --cov=${PROJECT_PACKAGE} --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated!$(NC)"

.PHONY: test-unit
test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	${UV} run pytest tests/unit/ -v
	@echo "$(GREEN)Unit tests completed!$(NC)"

.PHONY: test-integration
test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	${UV} run pytest tests/integration/ -v
	@echo "$(GREEN)Integration tests completed!$(NC)"

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	${UV} run pytest-watch tests/
	@echo "$(GREEN)Watch mode activated!$(NC)"

.PHONY: uv-test
uv-test: test-verbose ## Run all tests via uv with verbose output (alias for test-verbose)

# =============================================================================
# CODE QUALITY
# =============================================================================

.PHONY: lint
lint: ## Run linting with ruff
	@echo "$(BLUE)Running linter...$(NC)"
	${UV} run ruff check .
	@echo "$(GREEN)Linting completed!$(NC)"

.PHONY: lint-fix
lint-fix: ## Fix linting issues automatically
	@echo "$(BLUE)Fixing linting issues...$(NC)"
	${UV} run ruff check --fix .
	@echo "$(GREEN)Linting issues fixed!$(NC)"

.PHONY: format
format: ## Format code and organize imports with ruff
	@echo "$(BLUE)Formatting code and organizing imports...$(NC)"
	${UV} run ruff check --select I --fix .
	${UV} run ruff format .
	@echo "$(GREEN)Code formatted!$(NC)"

.PHONY: format-check
format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(NC)"
	${UV} run ruff format --check .
	@echo "$(GREEN)Format check completed!$(NC)"

.PHONY: fix-format
fix-format: format ## Fix formatting errors (alias for format)

.PHONY: mypy
mypy: mypy-clean ## Run type checking with mypy
	@echo "$(BLUE)Running type checker...$(NC)"
	${UV} run mypy .
	@echo "$(GREEN)Type checking completed!$(NC)"

.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	${UV} run pre-commit install
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

.PHONY: pre-commit-run
pre-commit-run: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	${UV} run pre-commit run --all-files
	@echo "$(GREEN)Pre-commit hooks completed!$(NC)"

.PHONY: pre-commit
pre-commit: pre-commit-install pre-commit-run ## Install and run pre-commit hooks on all files

.PHONY: uv-pre-commit
uv-pre-commit: pre-commit-run ## Run pre-commit hooks via uv (alias for pre-commit-run)

.PHONY: quality
quality: lint format-check mypy test ## Run all quality checks
	@echo "$(GREEN)All quality checks completed!$(NC)"

.PHONY: quality-fix
quality-fix: lint-fix format mypy test ## Run all quality checks and fix issues
	@echo "$(GREEN)Quality fixes completed!$(NC)"

# =============================================================================
# UTILITY COMMANDS
# =============================================================================

.PHONY: clean
clean: ## Clean temporary files and caches
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "$(GREEN)Cleanup completed!$(NC)"

.PHONY: clean-logs
clean-logs: ## Clean log files
	@echo "$(BLUE)Cleaning log files...$(NC)"
	find . -name "*.log" -type f -delete
	@echo "$(GREEN)Log files cleaned!$(NC)"

.PHONY: mypy-clean
mypy-clean: ## Clean mypy cache
	@echo "$(BLUE)Cleaning mypy cache...$(NC)"
	rm -rf .mypy_cache/ 2>/dev/null || true
	@echo "$(GREEN)Mypy cache cleaned!$(NC)"

.PHONY: reset
reset: clean install ## Reset everything and reinstall
	@echo "$(GREEN)Project reset completed!$(NC)"

.PHONY: status
status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo "$(GREEN)Python environment:$(NC)"
	@${UV} run python --version 2>/dev/null || echo "No UV environment found"
	@echo "$(GREEN)Installed packages:$(NC)"
	@${UV} run pip list 2>/dev/null | head -10 || echo "No packages found"


# =============================================================================
# DEVELOPMENT SHORTCUTS
# =============================================================================

.PHONY: dev
dev: install pre-commit-install ## Quick start development environment
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "$(YELLOW)Run 'make quick-test' to verify everything is working$(NC)"

.PHONY: quick-test
quick-test: lint test ## Quick quality check
	@echo "$(GREEN)Quick test completed!$(NC)"

.PHONY: full-check
full-check: quality test-coverage ## Full project check
	@echo "$(GREEN)Full project check completed!$(NC)"

.PHONY: ci
ci: quality test-coverage ## Run CI pipeline locally
	@echo "$(GREEN)CI pipeline completed!$(NC)"