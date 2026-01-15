"""GradientBoostingモデル"""
from sklearn.ensemble import GradientBoostingRegressor
from .base_model import BaseModel

class GradientBoostModel(BaseModel):
    def build(self, **params):
        """GradientBoostingモデル構築"""
        return GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 1.0),
            random_state=params.get('random_state', 42)
        )
    
    def suggest_hyperparameters(self, trial):
        """Optunaパラメータ提案"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }
    
    def get_model_type(self):
        """モデルタイプ取得"""
        return 'tree'