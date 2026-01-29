# models/__init__.py
from .model_factory import ModelFactory

# 利用側で "from models import ModelFactory" と書けるようにする
__all__ = ["ModelFactory"]
