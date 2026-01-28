# ML API - Production-Grade Machine Learning Service

A production-ready ML API service built with FastAPI, Polars, SQLAlchemy 2.0, and Google Cloud Storage for model training, hyperparameter tuning, and inference.

## Features

- **Data Splits**: CRUD operations for train/val/test splits with multiple strategies
- **Experiments**: Hyperparameter tuning with Optuna for CatBoost, XGBoost, and LightGBM
- **Model Registry**: Version management with staging/production promotion
- **Predictions**: Inference with preprocessing/postprocessing pipelines
- **Feature Importance**: SHAP explanations with fallback methods
- **Polars-First**: Primary dataframe operations use Polars with explicit pandas conversion
- **Observability**: Structured logging, request tracing, and Prometheus metrics
- **Background Jobs**: Async experiment execution with Arq (Redis-based)

## Tech Stack

- **API Framework**: FastAPI
- **Database**: PostgreSQL with async SQLAlchemy 2.0
- **Migrations**: Alembic
- **Data Processing**: Polars (primary), Pandas (boundary conversion only)
- **ML Libraries**: CatBoost (default), XGBoost, LightGBM
- **Hyperparameter Tuning**: Optuna
- **Storage**: Google Cloud Storage
- **Job Queue**: Arq (Redis-based)
- **CLI**: Typer
- **Package Manager**: uv (10-100x faster than pip)
- **Python**: 3.12

## Quick Start

### Prerequisites

- Python 3.12
- PostgreSQL 14+
- Redis 6+
- Google Cloud Platform account with GCS bucket
- [uv](https://github.com/astral-sh/uv) >= 0.7 - Fast Python package installer

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <repo-url>
cd ml-api

# Install dependencies (recommended)
make sync

# Or manually with uv
uv sync --extra dev

# Copy and configure environment
cp .env.example .env
vim .env
```

### Environment Configuration

Key settings in `.env`:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/mlapi

# GCS
GCS_BUCKET=my-ml-artifacts
GCS_PROJECT_ID=my-project
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Redis
REDIS_URL=redis://localhost:6379/0

# Model defaults
DEFAULT_MODEL_TYPE=catboost
DEFAULT_CLASSIFICATION_METRIC=auc
DEFAULT_REGRESSION_METRIC=rmse
```

### Run Migrations

```bash
make db-upgrade

# Or manually
uv run --with alembic alembic upgrade head
```

### Start the API

```bash
# Development (with auto-reload)
make dev

# Or manually
uv run ml-api serve --reload --port 8000

# Production
uv run ml-api serve --workers 4 --port 8000
```

### Start Background Worker

```bash
make worker

# Or manually
uv run --with arq arq ml_api.workers.worker.WorkerSettings
```

## Development Commands

### Setup & Installation

```bash
make sync             # Install dependencies with uv (recommended)
make install          # Production dependencies only
make install-dev      # Development dependencies
```

### Code Quality

```bash
make lint             # Lint with ruff
make format           # Format with black + ruff
make format-check     # Check formatting without changes
make check            # All checks (format + lint)
```

### Testing

```bash
make test             # Run tests with coverage
```

### Database

```bash
make db-upgrade       # Run migrations
make db-downgrade     # Rollback last migration
make db-migrate       # Create new migration (will prompt for message)
```

### Build & Clean

```bash
make build            # Build distribution packages
make clean            # Clean build artifacts
```

### Docker (Local Development)

```bash
make docker-build     # Build Docker image
make docker-run       # Run Docker container
docker-compose up     # Start full stack (Postgres + Redis + API)
```

### Info

```bash
make help             # Show all available commands
make info             # Show project information
make version          # Show current version
```

## API Endpoints

### Health & Meta

- `GET /healthz` - Basic health check
- `GET /readyz` - Readiness check (DB + GCS connectivity)
- `GET /version` - Version information
- `GET /metrics` - Prometheus metrics

### Data Splits

- `POST /v1/splits` - Create a new data split
- `GET /v1/splits/{split_id}` - Get split details
- `GET /v1/splits` - List splits (with filters & pagination)
- `PATCH /v1/splits/{split_id}` - Update split metadata
- `DELETE /v1/splits/{split_id}` - Delete split (soft delete)

### Experiments

- `POST /v1/experiments` - Create and start experiment
- `GET /v1/experiments/{experiment_id}` - Get experiment details
- `GET /v1/experiments` - List experiments
- `PATCH /v1/experiments/{experiment_id}` - Update experiment
- `DELETE /v1/experiments/{experiment_id}` - Delete experiment
- `GET /v1/experiments/{experiment_id}/trials` - List trials
- `POST /v1/experiments/{experiment_id}/cancel` - Cancel running experiment

### Models

- `GET /v1/models` - List models (filter by stage)
- `GET /v1/models/{model_id}` - Get model details
- `POST /v1/models/{model_id}/promote` - Promote to production
- `POST /v1/models/{model_id}/archive` - Archive model
- `DELETE /v1/models/{model_id}` - Delete model

### Predictions

- `POST /v1/predict` - Make predictions

### Feature Importance

- `POST /v1/importance` - Get feature importance

## Example API Calls

### Create Data Split

```bash
curl -X POST http://localhost:8000/v1/splits \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "customer_churn",
    "inline_data": [
      {"age": 25, "income": 50000, "churned": 0},
      {"age": 35, "income": 75000, "churned": 1}
    ],
    "split_strategy": "random",
    "split_params": {
      "train_ratio": 0.7,
      "val_ratio": 0.15,
      "test_ratio": 0.15,
      "seed": 42
    }
  }'
```

### Create Experiment

```bash
curl -X POST http://localhost:8000/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "split_id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Customer Churn Model v1",
    "target_column": "churned",
    "feature_columns": ["age", "income"],
    "task_type": "classification",
    "model_type": "catboost",
    "metric_name": "auc",
    "optuna_config": {
      "n_trials": 50,
      "timeout_seconds": 3600,
      "sampler": "TPE",
      "pruner": "MedianPruner",
      "seed": 42
    }
  }'
```

### Make Prediction

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "stage": "production",
    "instances": [
      {"age": 30, "income": 60000},
      {"age": 45, "income": 90000}
    ]
  }'
```

### Get Feature Importance

```bash
curl -X POST http://localhost:8000/v1/importance \
  -H "Content-Type: application/json" \
  -d '{
    "stage": "production",
    "max_samples": 1000,
    "method_preference": ["shap", "native"]
  }'
```

## CLI Usage

The `ml-api` CLI provides comprehensive server configuration:

```bash
# Development
uv run ml-api serve --reload --log-level debug

# Production
uv run ml-api serve --host 0.0.0.0 --port 8000 --workers 4

# With SSL
uv run ml-api serve \
  --host 0.0.0.0 \
  --port 8443 \
  --workers 4 \
  --ssl-keyfile /path/to/key.pem \
  --ssl-certfile /path/to/cert.pem

# With proxy headers (for reverse proxy)
uv run ml-api serve \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --proxy-headers \
  --forwarded-allow-ips="127.0.0.1,10.0.0.0/8"

# See all options
uv run ml-api serve --help
```

### CLI Options

- `--host, -h`: Bind host (default: 0.0.0.0)
- `--port, -p`: Bind port (default: 8000)
- `--workers, -w`: Number of worker processes (default: 1)
- `--reload, -r`: Enable auto-reload for development
- `--log-level, -l`: Set log level (debug, info, warning, error, critical)
- `--access-log/--no-access-log`: Enable/disable access logs
- `--proxy-headers`: Enable proxy headers for reverse proxy setups
- `--forwarded-allow-ips`: Comma-separated IPs to trust for proxy headers
- `--ssl-keyfile`: Path to SSL key file
- `--ssl-certfile`: Path to SSL certificate file
- `--limit-concurrency`: Maximum number of concurrent connections
- `--limit-max-requests`: Maximum requests before worker restart
- `--timeout-keep-alive`: Keep-alive timeout in seconds (default: 5)

## Release Process

### Automated Release (Tag-Based)

```bash
# 1. Bump version and create release
make release-patch    # 0.1.0 → 0.1.1
make release-minor    # 0.1.0 → 0.2.0
make release-major    # 0.1.0 → 1.0.0

# Or specify exact version
make release V=1.0.0

# This automatically:
# - Runs format-check, lint, and tests
# - Bumps version using uv version --bump (updates pyproject.toml and ml_api/__init__.py)
# - Creates git commit and tag
# - Pushes to GitHub

# 2. GitHub Actions automatically:
# - Runs full test suite
# - Builds package with uv
# - Publishes to PyPI
# - Creates GitHub release with changelog
```

### Manual Publishing (Testing)

```bash
# Publish to TestPyPI
make publish-test

# Publish to PyPI (manual - normally done via GitHub Actions)
make publish
```

### GitHub Secrets Required

Add to your GitHub repository settings → Secrets and variables → Actions:

- `PYPI_API_TOKEN` - Get from https://pypi.org/manage/account/token/

### CI/CD Workflows

#### Continuous Integration (ci.yml)
Triggers on push/PR to main/develop:
- Lint & format check
- Test suite with coverage (Python 3.12)
- Security scanning
- Package build

#### Release (publish.yml)
Triggers on tag push (v*.*.*):
- Full test suite
- Build package
- Publish to PyPI with `uv publish`
- Create GitHub release with auto-generated changelog

## Architecture

### Polars-First Data Flow

```
Input Data (JSON/CSV/Parquet)
    ↓ (Polars)
Data Split (train/val/test)
    ↓ (Polars)
Preprocessing (feature engineering, encoding)
    ↓ (Polars → Pandas conversion logged)
Model Training (CatBoost/XGBoost/LightGBM)
    ↓
Model Artifacts → GCS
    ↓
Prediction Request
    ↓ (Polars)
Preprocessing (apply saved artifacts)
    ↓ (Polars → Pandas for model)
Inference
    ↓ (Polars)
Postprocessing
    ↓
Response
```

**Key Principle**: Use Polars for all data manipulation. Convert to pandas ONLY when a specific ML library requires it, and log every conversion with memory estimates.

### GCS Artifact Layout

```
gs://{bucket}/
  splits/
    {split_id}/
      train.parquet
      val.parquet
      test.parquet
  experiments/
    {experiment_id}/
      trials/
        {trial_id}/
          model.cbm (or .json for XGBoost/LightGBM)
      best_model/
        model.cbm
        preprocess.json
        postprocess.json
        metrics.json
        signature.json
  models/
    {experiment_id}/
      v{version}/
        model.cbm
        preprocess.json
        postprocess.json
        signature.json
```

## Logging & Observability

### Structured Logging

All logs include:
- `request_id`: Correlation ID for request tracing
- `service`: Service name
- `environment`: Current environment
- Contextual fields: `entity_id`, `split_id`, `experiment_id`, `trial_id`, `model_id`

### Log Lifecycle Events

Major operations log:
- Start: `{operation}_started`
- Completion: `{operation}_completed` with duration
- Failure: `{operation}_failed` with error details

### Dataframe Logging

All dataframe operations log:
- Rows and columns
- Schema summary
- Memory estimates
- Conversions between Polars ↔ Pandas

### Metrics

Prometheus metrics available at `/metrics`:
- HTTP request counts and durations
- Job execution stats
- Prediction counts
- Training metrics

## Troubleshooting

### GCS Connection Issues

```bash
# Verify credentials
gcloud auth application-default login

# Check bucket permissions
gsutil ls gs://{bucket-name}

# Test with Python
python -c "from google.cloud import storage; client = storage.Client(); print(list(client.list_buckets()))"
```

### Database Connection Issues

```bash
# Test connection
psql $DATABASE_URL

# Check migrations
make db-upgrade

# Or manually
uv run --with alembic alembic current
uv run --with alembic alembic history
```

### Worker Issues

```bash
# Check Redis connection
redis-cli ping

# Monitor worker logs
uv run --with arq arq ml_api.workers.worker.WorkerSettings --verbose

# Check Redis queue
redis-cli KEYS "arq:*"
```

### Dependencies Out of Sync

```bash
make clean
make sync
```

## Adding a New Model Type

1. Create trainer in `app/services/training/{model}_trainer.py`
2. Implement `ModelTrainer` protocol
3. Add case to `dispatcher.py`
4. Add enum value to `ModelType` in `app/db/models/experiment.py`
5. Add tests in `tests/`

## Project Structure

```
ml-api/
├── app/                    # Main application code
│   ├── api/               # FastAPI routes
│   │   └── v1/           # API v1 endpoints
│   ├── core/             # Core functionality (config, deps, logging)
│   ├── db/               # Database models and session
│   │   └── models/       # SQLAlchemy models
│   ├── schemas/          # Pydantic models
│   ├── services/         # Business logic
│   │   ├── storage/      # GCS operations
│   │   ├── training/     # ML training services
│   │   └── inference/    # Prediction services
│   ├── workers/          # Background job workers
│   └── main.py           # FastAPI app entry point
├── cli/                   # CLI commands
│   └── main.py           # Typer CLI entry point
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── alembic/              # Database migrations
├── .github/              # GitHub Actions workflows
│   └── workflows/        # CI/CD workflows
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Local development stack
├── Makefile              # Development commands
├── pyproject.toml        # Project dependencies & config
└── .env.example          # Example environment variables
```

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and checks (`make check && make test`)
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Support

For issues, questions, or contributions:
- Create an issue: https://github.com/username/ml-api/issues
- Check CI/CD: https://github.com/username/ml-api/actions
