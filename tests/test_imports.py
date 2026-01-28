"""Test that all imports work correctly."""


def test_import_main():
    """Test main app import."""
    from ml_api.main import app

    assert app is not None


def test_import_config():
    """Test config import."""
    from ml_api.core.config import settings

    assert settings is not None
    assert settings.app_name == "ml-api"


def test_import_cli():
    """Test CLI import."""
    from ml_api.cli.main import app as cli_app

    assert cli_app is not None


def test_import_models():
    """Test database models import."""
    from ml_api.db.models.split import DataSplit
    from ml_api.db.models.experiment import Experiment
    from ml_api.db.models.model_registry import ModelRegistry

    assert DataSplit is not None
    assert Experiment is not None
    assert ModelRegistry is not None


def test_import_schemas():
    """Test schema imports."""
    from ml_api.schemas.split import DataSplitCreate
    from ml_api.schemas.experiment import ExperimentCreate

    assert DataSplitCreate is not None
    assert ExperimentCreate is not None
