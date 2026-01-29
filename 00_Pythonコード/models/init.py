from .ridge import RidgeModel
from .elastic_net import ElasticNetModel
from .random_forest import RandomForestModel
from .lightgbm_model import LightGBMModel
from .catboost_model import CatBoostModel
from .xgboost_model import XGBoostModel

MODEL_REGISTRY = {
    "ridge": RidgeModel,
    "elastic_net": ElasticNetModel,
    "random_forest": RandomForestModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "xgboost": XGBoostModel
}

def get_model(name):
    """モデルインスタンスを取得"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]()

def list_available_models():
    """利用可能なモデルのリスト"""
    available = []
    for name, model_class in MODEL_REGISTRY.items():
        model = model_class()
        if hasattr(model, 'available'):
            if model.available:
                available.append(name)
        else:
            available.append(name)
    return available