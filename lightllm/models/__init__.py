import os
import importlib
import inspect
from pathlib import Path


def auto_import_models():
    """
    Automatically imports all classes from model.py files in model directories
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = Path(base_dir)
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_file = model_dir / "model.py"
        if not model_file.exists():
            continue
        module_path = f"lightllm.models.{model_dir.name}.model"

        try:
            importlib.import_module(module_path)
        except:
            pass


auto_import_models()
from .registry import get_model
