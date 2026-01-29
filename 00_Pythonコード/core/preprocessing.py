import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

class EnhancedPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, use_interactions=False, use_polynomial=False):
        self.use_interactions = use_interactions
        self.use_polynomial = use_polynomial
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.numeric_cols_ = None  # 学習時に固定
        self._fitted = False

    def __getstate__(self):
        """ES: Asegurar que _fitted se guarde explícitamente en el estado serializado
        EN: Ensure _fitted is explicitly stored in the serialized state
        JA: シリアライズ状態に _fitted を明示的に保存する
        """
        state = self.__dict__.copy()
        # ES: Asegurar que _fitted esté en el estado (puede que no esté si se accedió vía __getattr__)
        # EN: Ensure _fitted is present in the state (it may be missing if accessed via __getattr__)
        # JA: _fitted が state に存在することを保証（__getattr__ 経由だと欠ける可能性）
        if '_fitted' not in state:
            # ES: Calcular _fitted basándose en el estado actual
            # EN: Compute _fitted based on the current state
            # JA: 現在の状態から _fitted を算出
            has_numeric_cols = state.get('numeric_cols_') is not None
            scaler = state.get('scaler')
            has_fitted_scaler = False
            if scaler is not None:
                try:
                    has_fitted_scaler = hasattr(scaler, 'scale_')
                except:
                    pass
            state['_fitted'] = has_numeric_cols or has_fitted_scaler
        return state

    def __setstate__(self, state):
        """ES: Compatibilidad con objetos guardados antiguos que no tienen _fitted o numeric_cols_
        EN: Compatibility for older saved objects missing _fitted or numeric_cols_
        JA: _fitted / numeric_cols_ が無い旧保存オブジェクトとの互換
        """
        # ES: Actualizar __dict__ primero
        # EN: Update __dict__ first
        # JA: まず __dict__ を更新
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            if hasattr(super(), '__setstate__'):
                super().__setstate__(state)
            else:
                self.__dict__.update(state)
        
        # ES: Asegurar que numeric_cols_ exista primero
        # EN: Ensure numeric_cols_ exists first
        # JA: numeric_cols_ が先に存在することを保証
        if 'numeric_cols_' not in self.__dict__:
            self.__dict__['numeric_cols_'] = None
        
        # ES: Asegurar que _fitted exista después de cargar (CRÍTICO: debe estar en __dict__)
        # EN: Ensure _fitted exists after loading (CRITICAL: must be in __dict__)
        # JA: 読み込み後に _fitted を必ず用意（重要：__dict__ に必要）
        # ES: Inicializar INMEDIATAMENTE después de actualizar __dict__ para prevenir errores
        # EN: Initialize IMMEDIATELY after updating __dict__ to prevent errors
        # JA: __dict__ 更新直後に即初期化してエラー防止
        if '_fitted' not in self.__dict__:
            # ES: Si tiene numeric_cols_ o los scalers están fitted, asumir que está fitted
            # EN: If numeric_cols_ exists or scalers are fitted, assume the object is fitted
            # JA: numeric_cols_ がある/スケーラがfit済みならfit済みとみなす
            has_numeric_cols = self.__dict__.get('numeric_cols_') is not None
            scaler = self.__dict__.get('scaler')
            has_fitted_scaler = False
            if scaler is not None:
                try:
                    has_fitted_scaler = hasattr(scaler, 'scale_')
                except:
                    pass
            # ES: CRÍTICO: Usar __dict__ directamente para evitar cualquier problema de acceso
            # EN: CRITICAL: Use __dict__ directly to avoid any attribute-access issues
            # JA: 重要：属性アクセス問題回避のため __dict__ を直接使用
            self.__dict__['_fitted'] = has_numeric_cols or has_fitted_scaler
    
    def __getattr__(self, name):
        """ES: Compatibilidad: retorna valores por defecto para atributos faltantes
        EN: Compatibility: return default values for missing attributes
        JA: 互換：欠落属性にデフォルト値を返す
        """
        if name == '_fitted':
            # ES: Si _fitted no existe, verificar si el objeto está realmente fitted
            # EN: If _fitted does not exist, check whether the object is actually fitted
            # JA: _fitted が無い場合、実際にfit済みか確認
            has_numeric_cols = hasattr(self, 'numeric_cols_') and self.numeric_cols_ is not None
            has_fitted_scaler = hasattr(self.scaler, 'scale_') if hasattr(self, 'scaler') else False
            fitted_value = has_numeric_cols or has_fitted_scaler
            # ES: Usar object.__setattr__ para evitar recursión
            # EN: Use object.__setattr__ to avoid recursion
            # JA: 再帰回避のため object.__setattr__ を使用
            object.__setattr__(self, '_fitted', fitted_value)
            return fitted_value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __getattribute__(self, name):
        """ES: Intercepta TODOS los accesos para asegurar que _fitted siempre exista
        EN: Intercept ALL accesses to ensure _fitted always exists
        JA: すべてのアクセスをフックして _fitted の存在を保証
        """
        # ES: Si estamos accediendo a _fitted, verificar primero si existe en __dict__
        # EN: If accessing _fitted, first check whether it exists in __dict__
        # JA: _fitted にアクセスする場合、まず __dict__ にあるか確認
        if name == '_fitted':
            # ES: Intentar acceder directamente a __dict__ sin usar super() para evitar recursión
            # EN: Try to read __dict__ directly (avoid super()) to prevent recursion
            # JA: 再帰回避のため super() を使わず __dict__ を直接参照
            try:
                # ES: Usar object.__getattribute__ para acceder a __dict__ sin recursión
                # EN: Use object.__getattribute__ to access __dict__ without recursion
                # JA: 再帰なしで __dict__ にアクセスするため object.__getattribute__ を使用
                d = object.__getattribute__(self, '__dict__')
                if '_fitted' in d:
                    return d['_fitted']
            except AttributeError:
                pass
            # ES: Si _fitted no existe en __dict__, crearlo basándose en el estado del objeto
            # EN: If _fitted is missing in __dict__, create it based on the object's state
            # JA: __dict__ に _fitted が無ければ状態から生成
            # ES: Usar __dict__ directamente para evitar recursión
            # EN: Use __dict__ directly to avoid recursion
            # JA: 再帰回避のため __dict__ を直接使用
            try:
                d = object.__getattribute__(self, '__dict__')
                has_numeric_cols = d.get('numeric_cols_') is not None
                scaler = d.get('scaler')
                has_fitted_scaler = False
                if scaler is not None:
                    try:
                        has_fitted_scaler = hasattr(scaler, 'scale_')
                    except:
                        pass
                fitted_value = has_numeric_cols or has_fitted_scaler
                d['_fitted'] = fitted_value
                return fitted_value
            except AttributeError:
                # ES: Si ni siquiera __dict__ existe, retornar False como fallback
                # EN: If __dict__ does not exist, return False as a fallback
                # JA: __dict__ 自体が無い場合はフォールバックで False
                return False
        
        # ES: Para otros atributos, usar el comportamiento normal
        # EN: For other attributes, use the normal behavior
        # JA: その他の属性は通常動作
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if name == '_fitted':
                # ES: Fallback adicional si el primer intento falló
                # EN: Additional fallback if the first attempt failed
                # JA: 追加フォールバック（最初が失敗した場合）
                return False
            raise

    def _extract_numeric(self, X):
        """学習時は列を確定、推論時は確定済み列で取り出す"""
        # ES: Asegurar que _fitted exista antes de cualquier operación
        # EN: Ensure _fitted exists before any operation
        # JA: どの処理よりも前に _fitted の存在を保証
        if '_fitted' not in self.__dict__:
            has_numeric_cols = hasattr(self, 'numeric_cols_') and self.numeric_cols_ is not None
            has_fitted_scaler = hasattr(self.scaler, 'scale_') if hasattr(self, 'scaler') else False
            self.__dict__['_fitted'] = has_numeric_cols or has_fitted_scaler
        
        if isinstance(X, pd.DataFrame):
            if self.numeric_cols_ is None:
                self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = X[self.numeric_cols_].values
        else:
            # ndarray の場合はそのまま
            X_numeric = np.asarray(X)
        return X_numeric

    def fit(self, X, y=None):
        """Expose sklearn-style fit so check_is_fitted treats this as an estimator."""
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        X_numeric = self._extract_numeric(X)
        X_imputed = self.imputer.fit_transform(X_numeric)
        X_scaled = self.scaler.fit_transform(X_imputed)
        self._fitted = True
        # 既存パイプラインに合わせて float32 で返す
        return np.asarray(X_scaled, dtype=np.float32)

    def transform(self, X):
        # ES: Asegurar que _fitted exista ANTES de cualquier acceso (CRÍTICO)
        # EN: Ensure _fitted exists BEFORE any access (CRITICAL)
        # JA: どのアクセスよりも前に _fitted の存在を保証（重要）
        if '_fitted' not in self.__dict__:
            has_numeric_cols = self.__dict__.get('numeric_cols_') is not None
            scaler = self.__dict__.get('scaler')
            has_fitted_scaler = hasattr(scaler, 'scale_') if scaler is not None else False
            self.__dict__['_fitted'] = has_numeric_cols or has_fitted_scaler
        
        # ES: Compatibilidad: verificar si está fitted sin acceder directamente a _fitted
        # EN: Compatibility: check fitted state without directly accessing _fitted
        # JA: 互換：_fitted を直接参照せずにfit状態を確認
        # ES: Primero intentar getattr, si falla verificar otros indicadores
        # EN: First try getattr; if it fails, check other indicators
        # JA: まず getattr を試し、失敗したら他の指標で確認
        try:
            is_fitted = getattr(self, '_fitted', None)
            if is_fitted is None:
                # ES: _fitted no existe, verificar otros indicadores
                # EN: _fitted is missing; check other indicators
                # JA: _fitted が無い場合は他の指標で確認
                has_numeric_cols = hasattr(self, 'numeric_cols_') and self.numeric_cols_ is not None
                has_fitted_scaler = hasattr(self.scaler, 'scale_')
                is_fitted = has_numeric_cols or has_fitted_scaler
                # ES: Guardar para futuros accesos
                # EN: Persist for future accesses
                # JA: 次回以降のため保存
                self.__dict__['_fitted'] = is_fitted
            elif not is_fitted:
                # ES: _fitted existe pero es False, verificar otros indicadores como fallback
                # EN: _fitted exists but is False; verify other indicators as a fallback
                # JA: _fitted が False の場合、フォールバックで他指標を確認
                has_numeric_cols = hasattr(self, 'numeric_cols_') and self.numeric_cols_ is not None
                has_fitted_scaler = hasattr(self.scaler, 'scale_')
                if has_numeric_cols or has_fitted_scaler:
                    is_fitted = True
                    self.__dict__['_fitted'] = True
        except AttributeError:
            # ES: Si getattr falla completamente, verificar otros indicadores
            # EN: If getattr fails completely, check other indicators
            # JA: getattr が完全に失敗した場合、他指標で確認
            has_numeric_cols = hasattr(self, 'numeric_cols_') and self.numeric_cols_ is not None
            has_fitted_scaler = hasattr(self.scaler, 'scale_')
            is_fitted = has_numeric_cols or has_fitted_scaler
            self.__dict__['_fitted'] = is_fitted
        
        if not is_fitted:
            raise RuntimeError("EnhancedPreprocessor is not fitted. Call fit_transform() first.")
        
        X_numeric = self._extract_numeric(X)
        X_imputed = self.imputer.transform(X_numeric)
        X_scaled = self.scaler.transform(X_imputed)
        return np.asarray(X_scaled, dtype=np.float32)

    # 既存パイプラインがあれば使うフック（情報公開のみ）
    def get_feature_names(self):
        return list(self.numeric_cols_) if self.numeric_cols_ is not None else None


class AdvancedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=30, corr_threshold=0.95, use_mutual_info=False,
                 use_correlation_removal=True, mandatory_features=None,
                 use_mandatory_features=True, feature_names=None):
        self.top_k = int(top_k)
        self.corr_threshold = float(corr_threshold)
        # ES: CRÍTICO: Debe asignarse para compatibilidad
        # EN: CRITICAL: Must be assigned for compatibility
        # JP: 重要: 互換性のため必ず代入する必要がある
        self.use_mutual_info = use_mutual_info
        self.use_correlation_removal = use_correlation_removal
        self.mandatory_features = mandatory_features or []
        self.use_mandatory_features = use_mandatory_features
        self.feature_names = feature_names
        self.selected_features_ = None

    def __setstate__(self, state):
        """Compatibilidad con objetos guardados antiguos que no tienen use_mutual_info"""
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            if hasattr(super(), '__setstate__'):
                super().__setstate__(state)
            else:
                self.__dict__.update(state)
        # ES: Asegurar que use_mutual_info siempre exista después de cargar
        # EN: Ensure use_mutual_info always exists after loading
        # JP: 読み込み後にuse_mutual_infoが必ず存在するようにする
        if 'use_mutual_info' not in self.__dict__:
            self.__dict__['use_mutual_info'] = False

    def fit(self, X, y):
        """Provide sklearn-compatible fit for new check_is_fitted behavior."""
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y):
        X = np.asarray(X)
        n_features = X.shape[1]

        # 必須特徴量のインデックス
        mandatory_indices = []
        if self.use_mandatory_features and self.feature_names is not None:
            mandatory_indices = self._get_mandatory_indices()
        mandatory_indices = np.unique(mandatory_indices).astype(int) if len(mandatory_indices) else np.array([], dtype=int)

        # top_k を超える必須がある場合は、まず必須を優先し、残り枠を0に
        remaining_slots = max(0, self.top_k - len(mandatory_indices))

        if n_features <= self.top_k:
            # そのまま全部
            candidate_features = np.arange(n_features, dtype=int)
        else:
            # 1) 重要度でソート
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_

            if len(mandatory_indices) > 0:
                # 必須以外から残り枠を選択
                mask = np.ones(n_features, dtype=bool)
                mask[mandatory_indices] = False
                non_mand_idx = np.where(mask)[0]
                if remaining_slots > 0 and len(non_mand_idx) > 0:
                    non_mand_imp = importances[non_mand_idx]
                    pick = non_mand_idx[np.argsort(non_mand_imp)[::-1][:remaining_slots]]
                    candidate_features = np.unique(np.concatenate([mandatory_indices, pick])).astype(int)
                else:
                    candidate_features = mandatory_indices
            else:
                # 必須なし
                candidate_features = np.argsort(importances)[::-1][:self.top_k].astype(int)

            # 2) 相関除去（必要なら）
            if self.use_correlation_removal and len(candidate_features) > 1:
                candidate_features = self._remove_correlated_features(
                    X, candidate_features, importances, mandatory_indices
                )

        self.selected_features_ = np.asarray(candidate_features, dtype=int)
        return X[:, self.selected_features_]

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("AdvancedFeatureSelector is not fitted. Call fit_transform() first.")
        X = np.asarray(X)
        return X[:, self.selected_features_]

    def _get_mandatory_indices(self):
        if not self.mandatory_features or not self.feature_names:
            return []
        idxs = []
        for name in self.mandatory_features:
            if name in self.feature_names:
                idxs.append(self.feature_names.index(name))
            else:
                # 見つからない場合は通知のみ（落とさない）
                print(f"⚠ 必須特徴量 '{name}' が見つかりません")
        return idxs

    def _remove_correlated_features(self, X, candidate_features, importances, mandatory_indices=None):
        if mandatory_indices is None:
            mandatory_indices = np.array([], dtype=int)
        cand = np.asarray(candidate_features, dtype=int)
        Xc = X[:, cand]
        # 相関行列（NaN は 0 とみなす）
        corr = np.corrcoef(Xc.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        to_remove = set()
        L = len(corr)
        for i in range(L):
            for j in range(i+1, L):
                if abs(corr[i, j]) >= self.corr_threshold:
                    f1, f2 = cand[i], cand[j]
                    # 必須は除去しない
                    if f1 in mandatory_indices or f2 in mandatory_indices:
                        continue
                    # 重要度の低い方を落とす
                    if importances[f1] < importances[f2]:
                        to_remove.add(f1)
                    else:
                        to_remove.add(f2)
        keep = [f for f in cand if f not in to_remove]
        return np.asarray(keep, dtype=int)
