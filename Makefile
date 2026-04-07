.PHONY: setup install run api test eval clean lint format help

# Colors for terminal output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo '$(BLUE)Financial Knowledge Assistant$(NC)'
	@echo '=============================='
	@echo ''
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create virtual environment and install dependencies
	@echo '$(BLUE)Creating virtual environment...$(NC)'
	python -m venv venv
	@echo '$(BLUE)Installing dependencies...$(NC)'
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo '$(GREEN)Setup complete! Activate with: source venv/bin/activate$(NC)'

setup-dev: setup ## Setup with development dependencies
	@echo '$(BLUE)Installing development dependencies...$(NC)'
	./venv/bin/pip install -r requirements-dev.txt
	@echo '$(GREEN)Development setup complete!$(NC)'

install: ## Install dependencies (assumes venv is active)
	pip install --upgrade pip
	pip install -r requirements.txt

run: ## Run Streamlit application
	@echo '$(BLUE)Starting Streamlit app...$(NC)'
	streamlit run ui/streamlit_app.py --server.port 8501

api: ## Run FastAPI backend
	@echo '$(BLUE)Starting FastAPI server...$(NC)'
	uvicorn app.api.routes:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests
	@echo '$(BLUE)Running tests...$(NC)'
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	@echo '$(BLUE)Running tests with coverage...$(NC)'
	pytest tests/ --cov=app --cov-report=html --cov-report=term-missing
	@echo '$(GREEN)Coverage report: htmlcov/index.html$(NC)'

eval: ## Run RAG evaluation
	@echo '$(BLUE)Running evaluation pipeline...$(NC)'
	python scripts/evaluate_rag.py

ingest: ## Ingest sample documents
	@echo '$(BLUE)Ingesting documents...$(NC)'
	python scripts/ingest_documents.py --source data/sample_docs/

lint: ## Run linters
	@echo '$(BLUE)Running linters...$(NC)'
	ruff check app/ ui/ tests/
	mypy app/ --ignore-missing-imports

format: ## Format code
	@echo '$(BLUE)Formatting code...$(NC)'
	black app/ ui/ tests/
	isort app/ ui/ tests/
	@echo '$(GREEN)Code formatted!$(NC)'

clean: ## Clean up generated files
	@echo '$(BLUE)Cleaning up...$(NC)'
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage 2>/dev/null || true
	@echo '$(GREEN)Cleaned!$(NC)'

clean-data: ## Clean vector database and processed data
	@echo '$(YELLOW)Warning: This will delete all indexed data!$(NC)'
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && \
		rm -rf data/chroma_db data/processed/* && \
		echo '$(GREEN)Data cleaned!$(NC)' || \
		echo 'Cancelled.'

docker-build: ## Build Docker image
	@echo '$(BLUE)Building Docker image...$(NC)'
	docker-compose build

docker-run: ## Run with Docker Compose
	@echo '$(BLUE)Starting containers...$(NC)'
	docker-compose up -d
	@echo '$(GREEN)App running at http://localhost:8501$(NC)'

docker-stop: ## Stop Docker containers
	@echo '$(BLUE)Stopping containers...$(NC)'
	docker-compose down
