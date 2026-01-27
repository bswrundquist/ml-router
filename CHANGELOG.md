# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and core functionality

## [0.1.0] - 2024-01-26

### Added
- ✨ **Polars-First Data Processing**: Complete data I/O and preprocessing using Polars with explicit pandas conversion logging
- 🏗️ **SQLAlchemy 2.0 Async ORM**: Database models for DataSplit, Experiment, Trial, and ModelRegistry
- 🚀 **FastAPI Application**: Production-ready API with structured logging and request tracing
- 📊 **Model Training**: Support for CatBoost (default), XGBoost, and LightGBM
- 🔧 **Hyperparameter Tuning**: Optuna integration with customizable search spaces
- 🗄️ **GCS Artifact Management**: Versioned model artifacts with deterministic paths
- 📝 **Data Split Strategies**: Random, time-based, and entity-based splitting
- 🎯 **Match/Case Dispatch**: Python 3.10+ pattern matching for model and task type routing
- 📈 **Observability**: Structured logging with request_id correlation and Prometheus metrics
- 🔒 **Type Safety**: Full Pydantic validation and SQLAlchemy 2.0 type hints
- 🐳 **Docker Support**: Complete docker-compose stack for local development
- 📚 **Comprehensive Documentation**: README, implementation guide, cURL examples, and project summary
- 🧪 **Testing Infrastructure**: Pytest fixtures with fake GCS client

### API Endpoints
- Health & Meta: `/healthz`, `/readyz`, `/version`, `/metrics`
- Data Splits: Full CRUD at `/v1/splits`

### Infrastructure
- Alembic migrations setup
- Pydantic Settings for environment-based configuration
- Custom exception handling with consistent error responses
- Request ID middleware for tracing
- Prometheus metrics collection

### Documentation
- Complete README with installation, usage, and troubleshooting
- CURL_EXAMPLES.md with 50+ API examples
- IMPLEMENTATION_GUIDE.md with file-by-file breakdown
- PROJECT_SUMMARY.md with architecture overview

### Development
- uv for fast dependency management
- Black + Ruff for code formatting and linting
- mypy for type checking
- pytest for testing
- GitHub Actions CI/CD workflows powered by uv

## Release Notes

### 0.1.0 - Initial Beta Release

This is the initial beta release of the ML API service. It provides a solid foundation for machine learning workflows with:

- **Production-grade architecture** with async database, GCS storage, and background job support
- **Polars-first data processing** for superior performance
- **Multi-model support** with easy extensibility
- **Complete observability** with structured logging and metrics
- **Type-safe codebase** with Pydantic and SQLAlchemy 2.0

#### What's Working
- ✅ Data split creation and management
- ✅ GCS artifact storage
- ✅ Core training infrastructure
- ✅ Model trainer implementations (CatBoost, XGBoost, LightGBM)
- ✅ Health and metrics endpoints

#### Coming Soon
- 🔜 Experiment CRUD endpoints
- 🔜 Model registry management
- 🔜 Prediction service
- 🔜 SHAP feature importance
- 🔜 Complete CLI tools
- 🔜 Background job workers

[Unreleased]: https://github.com/yourusername/ml-router/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/ml-router/releases/tag/v0.1.0
