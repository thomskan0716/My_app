"""
モデルファクトリー - CatBoost + Gaussian Process 統合版
"""
# 先頭の相対インポート群を次のように書き換え（または追記）

try:
    from .ridge_model import RidgeModel
    from .elastic_net_model import ElasticNetModel
    from .random_forest_model import RandomForestModel
except Exception:
    # パッケージ構成が無い配布先向けフォールバック（同ディレクトリ直下にある想定）
    from ridge_model import RidgeModel
    from elastic_net_model import ElasticNetModel
    from random_forest_model import RandomForestModel

try:
    from .lasso_model import LassoModel
    LASSO_AVAILABLE = True
except ImportError:
    LASSO_AVAILABLE = False
    LassoModel = None

try:
    from .gradient_boost_model import GradientBoostModel
    GRADBOOST_AVAILABLE = True
except ImportError:
    GRADBOOST_AVAILABLE = False
    GradientBoostModel = None

try:
    from .lightgbm_model import LightGBMModel
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LightGBMModel = None

try:
    from .xgboost_model import XGBoostModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBoostModel = None

try:
    from .catboost_model import CatBoostModel
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostModel = None

try:
    from .gaussian_process_model import GaussianProcessModel
    GAUSSIANPROCESS_AVAILABLE = True
except ImportError:
    GAUSSIANPROCESS_AVAILABLE = False
    GaussianProcessModel = None


class ModelFactory:
    """モデル生成ファクトリー"""
    
    @staticmethod
    def get_available_models():
        models = {}
    
        # 基本モデル
        models['ridge'] = RidgeModel
        models['elastic_net'] = ElasticNetModel
        models['elasticnet'] = ElasticNetModel
        models['random_forest'] = RandomForestModel
        models['randomforest'] = RandomForestModel
    
        # Lasso
        if LASSO_AVAILABLE and LassoModel is not None:
            models['lasso'] = LassoModel
    
        # GradientBoost
        if GRADBOOST_AVAILABLE and GradientBoostModel is not None:
            models['gradient_boost'] = GradientBoostModel
            models['gradientboost'] = GradientBoostModel
    
        # LightGBM
        if LIGHTGBM_AVAILABLE and LightGBMModel is not None:
            try:
                if getattr(LightGBMModel(), 'available', True):
                    models['lightgbm'] = LightGBMModel
            except Exception as e:
                print(f"⚠ LightGBM initialization failed: {e}")
    
        # XGBoost
        if XGBOOST_AVAILABLE and XGBoostModel is not None:
            try:
                if getattr(XGBoostModel(), 'available', True):
                    models['xgboost'] = XGBoostModel
            except Exception as e:
                print(f"⚠ XGBoost initialization failed: {e}")
    
        # CatBoost
        if CATBOOST_AVAILABLE and CatBoostModel is not None:
            try:
                if getattr(CatBoostModel(), 'available', True):
                    models['catboost'] = CatBoostModel
            except Exception as e:
                print(f"⚠ CatBoost initialization failed: {e}")
    
        # Gaussian Process
        if GAUSSIANPROCESS_AVAILABLE and GaussianProcessModel is not None:
            try:
                if getattr(GaussianProcessModel(), 'available', True):
                    models['gaussian_process'] = GaussianProcessModel
                    models['gp'] = GaussianProcessModel
            except Exception as e:
                print(f"⚠ Gaussian Process initialization failed: {e}")
    
        return models

    
    @staticmethod
    def create_model(model_name):
        models = ModelFactory.get_available_models()
        key = str(model_name).lower().strip()   # ← 正規化を追加
    
        if key not in models:
            # エイリアスを除いた一覧を作成（重複除去）
            alias_exclude = {'elasticnet', 'randomforest', 'gradientboost', 'gp'}
            available = sorted({k for k in models.keys() if k not in alias_exclude})
            raise ValueError(
                f"Unknown model: '{model_name}'. Available models: {available}"
            )
        return models[key]()                     # ← 正規化した key を使用

    
    @staticmethod
    def list_models(verbose=False):
        """
        利用可能なモデルをリスト表示
        
        Args:
            verbose: 詳細情報を表示するか
        """
        models = ModelFactory.get_available_models()
        
        # エイリアスを除外した一意のモデルリスト
        unique_models = {}
        for name, model_class in models.items():
            if name not in ['elasticnet', 'randomforest', 'gradientboost', 'gp']:
                unique_models[name] = model_class
        
        if verbose:
            print("="*50)
            print("利用可能なモデル:")
            print("="*50)
            for name, model_class in sorted(unique_models.items()):
                try:
                    instance = model_class()
                    status = "✅ Available"
                    if hasattr(instance, 'available'):
                        status = "✅ Available" if instance.available else "❌ Not Available"
                    print(f"  {name:20} : {status}")
                except Exception as e:
                    print(f"  {name:20} : ❌ Error - {str(e)[:30]}")
            print("="*50)
        
        return sorted(unique_models.keys())


# デバッグ用：モジュールインポート時に利用可能なモデルを表示
if __name__ == "__main__":
    ModelFactory.list_models(verbose=True)