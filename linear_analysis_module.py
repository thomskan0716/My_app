



#!/usr/bin/env python
# coding: utf-8

"""
ES: Módulo de Análisis Lineal para 0.00sec.
EN: Linear analysis module for 0.00sec.
JA: 0.00sec 用の線形解析モジュール。

ES: Basado en 線形モデル_回帰分離混合_Ver2_noA11A21A32.py.
EN: Based on 線形モデル_回帰分離混合_Ver2_noA11A21A32.py.
JA: 線形モデル_回帰分離混合_Ver2_noA11A21A32.py をベースにしています。

ES: Adaptado para trabajar con la base de datos del proyecto.
EN: Adapted to work with the project's database.
JA: プロジェクトのDBで動くように調整されています。
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime

# 統計・機械学習
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score
)

# 統計的検定
from scipy import stats
from scipy.stats import shapiro, boxcox

# 可視化
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

warnings.filterwarnings('ignore')

class LinearAnalysisConfig:
    """ES: Configuración del análisis lineal
    EN: Linear analysis configuration
    JA: 線形解析の設定
    """
    
    # ES: Columnas objetivo (variables dependientes)
    # EN: Target columns (dependent variables)
    # JA: 目的変数列（従属変数）
    TARGET_COLUMNS = ['バリ除去', '摩耗量', '上面ダレ量', '側面ダレ量']
    
    # ES: Tipos de tarea para cada objetivo
    # EN: Task type per target
    # JA: 目的変数ごとのタスク種別
    TARGET_TYPES = {
        'バリ除去': 'classification',
        '摩耗量': 'regression',
        '上面ダレ量': 'regression', 
        '側面ダレ量': 'regression'
    }
    
    # ES: Columnas de características (variables independientes)
    # EN: Feature columns (independent variables)
    # JA: 特徴量列（独立変数）
    # ES: Mapeo de nombres de la BD a nombres del análisis
    # EN: Map DB column names to analysis column names
    # JA: DB列名→解析列名のマッピング
    FEATURE_COLUMNS = [
        '送り速度', 'UPカット', '切込量', 
        '突出量', '載せ率', '回転速度', 'パス数'  # Fixed: DB uses '突出量'
    ]
    
    # ES: Mapeo de nombres de la BD a nombres del análisis
    # EN: Map DB column names to analysis column names
    # JA: DB列名→解析列名のマッピング
    # ES: Nota: La BD usa '突出量' pero el análisis espera '突出量' (sin し)
    # EN: Note: the DB uses '突出量' and the analysis expects '突出量' (no し)
    # JA: 注意：DBは「突出量」、解析も「突出量」（し無し）を期待
    DB_TO_ANALYSIS_MAPPING = {
        '送り速度': '送り速度',
        'UPカット': 'UPカット', 
        '切込量': '切込量',
        '突出量': '突出量',  # Fixed: DB uses '突出量'
        '載せ率': '載せ率',
        '回転速度': '回転速度',
        'パス数': 'パス数'
    }
    
    # Mapeo inverso
    ANALYSIS_TO_DB_MAPPING = {v: k for k, v in DB_TO_ANALYSIS_MAPPING.items()}
    
    INNER_CV_SPLITS = 5
    OUTER_CV_SPLITS = 5
    RANDOM_STATE = 42

class LinearAnalysisPipeline:
    """ES: Pipeline de análisis lineal simplificado
    EN: Simplified linear analysis pipeline
    JA: 簡易線形解析パイプライン
    """
    
    def __init__(self, output_dir: str = "output_analysis"):
        """ES: Inicializar el pipeline
        EN: Initialize the pipeline
        JA: パイプラインを初期化
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.transformation_info = {}
        
        # ES: Configurar matplotlib para japonés
        # EN: Configure matplotlib for Japanese fonts
        # JA: matplotlib を日本語フォント向けに設定
        self._setup_japanese_font()
    
    def _setup_japanese_font(self):
        """ES: Configurar fuente japonesa para matplotlib
        EN: Configure Japanese font for matplotlib
        JA: matplotlib の日本語フォント設定
        """
        try:
            if os.name == 'nt':
                fonts = ['MS Gothic', 'Yu Gothic', 'Meiryo']
            else:
                fonts = ['IPAexGothic', 'Hiragino Sans', 'Noto Sans CJK JP']
            
            for font in fonts:
                try:
                    mpl.rcParams['font.family'] = font
                    mpl.rcParams['font.size'] = 12
                    break
                except:
                    continue
        except Exception as e:
            print(f"⚠️ 日本語フォントを設定できませんでした: {e}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ES: Preparar datos para el análisis
        EN: Prepare data for analysis
        JA: 解析用データを準備"""
        print("🔧 解析用データを準備中...")
        
        # ES: Mapear nombres de columnas de la BD a nombres del análisis
        # EN: Map DB column names to analysis names
        # JA: DB列名を解析名にマッピング
        column_mapping = {}
        for db_col, analysis_col in LinearAnalysisConfig.DB_TO_ANALYSIS_MAPPING.items():
            if db_col in df.columns:
                column_mapping[db_col] = analysis_col
        
        # ES: Crear DataFrame con nombres mapeados
        # EN: Create a DataFrame with mapped names
        # JP: マッピング済み名称でDataFrameを作成
        df_mapped = df.rename(columns=column_mapping)
        
        # ES: Seleccionar solo las columnas necesarias
        # EN: Select only the required columns
        # JP: 必要な列のみ選択
        available_features = [col for col in LinearAnalysisConfig.FEATURE_COLUMNS 
                            if col in df_mapped.columns]
        available_targets = [col for col in LinearAnalysisConfig.TARGET_COLUMNS 
                           if col in df_mapped.columns]
        
        print(f"🔧 利用可能な特徴量: {available_features}")
        print(f"🔧 利用可能な目的変数: {available_targets}")
        
        # ES: Crear X (características) e y (objetivos) | EN: Build X (features) and y (targets) | JA: X（特徴量）とy（目的）を作成
        X = df_mapped[available_features].copy()
        y = df_mapped[available_targets].copy()
        
        # Manejar valores faltantes
        X = X.fillna(X.median())
        for col in y.columns:
            if y[col].dtype in ['int64', 'float64']:
                y[col] = y[col].fillna(y[col].median())
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.DataFrame):
        """Entrenar modelos para cada objetivo"""
        print("🔧 モデルを学習中...")
        
        for target_col in y.columns:
            if target_col not in LinearAnalysisConfig.TARGET_TYPES:
                continue
                
            task_type = LinearAnalysisConfig.TARGET_TYPES[target_col]
            print(f"🔧 {target_col} のモデルを学習中（{task_type}）")
            
            try:
                # ES: Obtener datos válidos para este objetivo | EN: Get valid data for this target | JA: この目的変数用の有効データを取得
                valid_mask = ~y[target_col].isnull()
                X_valid = X[valid_mask]
                y_valid = y[target_col][valid_mask]
                
                if len(X_valid) < 10:
                    print(f"⚠️ データ不足: {target_col}（{len(X_valid)} サンプル）")
                    continue
                
                if task_type == 'regression':
                    model_info = self._train_regression_model(X_valid, y_valid, target_col)
                else:
                    model_info = self._train_classification_model(X_valid, y_valid, target_col)
                
                self.models[target_col] = model_info
                
            except Exception as e:
                print(f"❌ モデル学習中にエラー: {target_col}: {e}")
                self.models[target_col] = {'error': str(e)}
    
    def _train_regression_model(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict:
        """ES: Entrenar modelo de regresión
        EN: Train regression model
        JA: 回帰モデルを学習"""
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=LinearAnalysisConfig.RANDOM_STATE),
            'Lasso': Lasso(random_state=LinearAnalysisConfig.RANDOM_STATE, max_iter=2000)
        }
        
        best_model_name = None
        best_score = -float('inf')
        best_model = None
        
        # ES: Validación cruzada simple | EN: Simple cross-validation | JA: 簡易交差検証
        cv = KFold(n_splits=LinearAnalysisConfig.INNER_CV_SPLITS, 
                   shuffle=True, random_state=LinearAnalysisConfig.RANDOM_STATE)
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                mean_score = scores.mean()
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
                    best_model = model
                    
            except Exception as e:
                print(f"⚠️ {name} でエラー: {e}")
        
        if best_model is None:
            best_model = LinearRegression()
            best_model_name = 'LinearRegression'
        
        # ES: Entrenar el mejor modelo
        # EN: Train the best model
        # JP: 最良モデルを学習
        best_model.fit(X, y)
        y_pred = best_model.predict(X)
        
        # Métricas finales
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # ES: Guardar modelo
        # EN: Save the model
        # JP: モデルを保存
        model_path = self.output_dir / f'model_{target_name}.pkl'
        model_data = {
            'model': best_model,
            'feature_names': X.columns.tolist(),
            'target_name': target_name,
            'model_name': best_model_name
        }
        joblib.dump(model_data, model_path)
        
        # ES: Crear gráfico | EN: Create chart | JA: グラフを作成
        self._plot_regression_results(y, y_pred, target_name)
        
        return {
            'model': best_model,
            'model_name': best_model_name,
            'model_path': str(model_path),
            'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
            'task_type': 'regression'
        }
    
    def _train_classification_model(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict:
        """ES: Entrenar modelo de clasificación
        EN: Train classification model
        JA: 分類モデルを訓練"""
        # ES: Verificar que hay suficientes muestras por clase
        # EN: Check that there are enough samples per class
        # JP: クラスごとのサンプル数が十分か確認する
        class_counts = y.value_counts()
        if len(class_counts) < 2 or class_counts.min() < 5:
            return {'error': 'insufficient_samples'}
        
        # Codificar etiquetas
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # ES: Entrenar modelo
        # EN: Train the model
        # JP: モデルを学習
        model = LogisticRegression(random_state=LinearAnalysisConfig.RANDOM_STATE, max_iter=2000)
        model.fit(X, y_encoded)
        
        # Predicciones
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # Métricas
        accuracy = accuracy_score(y_encoded, y_pred)
        f1 = f1_score(y_encoded, y_pred, average='weighted')
        
        # ES: Guardar modelo
        # EN: Save the model
        # JP: モデルを保存
        model_path = self.output_dir / f'model_{target_name}.pkl'
        model_data = {
            'model': model,
            'label_encoder': le,
            'feature_names': X.columns.tolist(),
            'target_name': target_name
        }
        joblib.dump(model_data, model_path)
        
        return {
            'model': model,
            'label_encoder': le,
            'model_name': 'LogisticRegression',
            'model_path': str(model_path),
            'metrics': {'accuracy': accuracy, 'f1_score': f1},
            'task_type': 'classification'
        }
    
    def _plot_regression_results(self, y_true: pd.Series, y_pred: np.ndarray, target_name: str):
        """ES: Crear gráfico de resultados de regresión
        EN: Create regression results plot
        JA: 回帰結果のグラフを作成"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # ES: Gráfico 1: Predicción vs Real | EN: Chart 1: Prediction vs Actual | JA: グラフ1：予測vs実測
            ax1.scatter(y_true, y_pred, alpha=0.6)
            ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax1.set_xlabel('Valor Real')
            ax1.set_ylabel('Predicción')
            ax1.set_title(f'{target_name}: Predicción vs Real')
            ax1.grid(True, alpha=0.3)
            
            # ES: Gráfico 2: Residuales | EN: Chart 2: Residuals | JA: グラフ2：残差
            residuals = y_true - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6)
            ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('Predicción')
            ax2.set_ylabel('Residuales')
            ax2.set_title(f'{target_name}: Análisis de Residuales')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # ES: Guardar gráfico | EN: Save chart | JA: グラフを保存
            plot_path = self.output_dir / f'regression_{target_name}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ グラフを保存しました: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ グラフ作成中にエラー: {target_name}: {e}")
    
    def save_results(self):
        """ES: Guardar resultados del análisis
        EN: Save analysis results
        JA: 解析結果を保存"""
        print("🔧 結果を保存中...")
        
        # Resumen de resultados
        results_summary = []
        for target_col, model_info in self.models.items():
            if 'error' in model_info:
                row = {
                    'Target': target_col,
                    'Status': 'Failed',
                    'Error': model_info['error']
                }
            else:
                row = {
                    'Target': target_col,
                    'Status': 'Success',
                    'Model': model_info['model_name'],
                    'Task_Type': model_info['task_type']
                }
                
                if 'metrics' in model_info:
                    metrics = model_info['metrics']
                    if model_info['task_type'] == 'regression':
                        row.update({
                            'MAE': f"{metrics.get('mae', 'N/A'):.4f}",
                            'RMSE': f"{metrics.get('rmse', 'N/A'):.4f}",
                            'R2': f"{metrics.get('r2', 'N/A'):.4f}"
                        })
                    else:
                        row.update({
                            'Accuracy': f"{metrics.get('accuracy', 'N/A'):.4f}",
                            'F1_Score': f"{metrics.get('f1_score', 'N/A'):.4f}"
                        })
            
            results_summary.append(row)
        
        # ES: Guardar como Excel
        # EN: Save as Excel
        # JP: Excelとして保存
        results_df = pd.DataFrame(results_summary)
        results_path = self.output_dir / 'analysis_results.xlsx'
        results_df.to_excel(results_path, index=False)
        print(f"✅ 結果を保存しました: {results_path}")
        
        # ES: Guardar como JSON
        # EN: Save as JSON
        # JP: JSONとして保存
        results_json = {
            'timestamp': datetime.now().isoformat(),
            'models': {k: {
                'model_name': v.get('model_name', 'Unknown'),
                'task_type': v.get('task_type', 'Unknown'),
                'metrics': v.get('metrics', {}),
                'error': v.get('error', None)
            } for k, v in self.models.items()},
            'output_directory': str(self.output_dir)
        }
        
        json_path = self.output_dir / 'analysis_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        print(f"✅ 結果JSONを保存しました: {json_path}")
    
    def run_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """ES: Ejecutar análisis completo
        EN: Run full analysis
        JA: 解析を一通り実行"""
        print("🚀 線形解析を開始...")
        
        try:
            # Preparar datos
            X, y = self.prepare_data(df)
            
            if X.empty or y.empty:
                raise ValueError("解析に使用できる有効データがありません")
            
            print(f"✅ データ準備完了: {X.shape[0]} サンプル, {X.shape[1]} 特徴量")
            
            # Entrenar modelos
            self.train_models(X, y)
            
            # ES: Guardar resultados
            # EN: Save results
            # JP: 結果を保存
            self.save_results()
            
            print("✅ 線形解析が正常に完了しました")
            
            return {
                'success': True,
                'models': self.models,
                'output_directory': str(self.output_dir),
                'data_shape': X.shape
            }
            
        except Exception as e:
            print(f"❌ 線形解析中にエラー: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def run_linear_analysis_from_db(db_manager, filters: Dict = None) -> Dict[str, Any]:
    """ES: Función principal para ejecutar análisis lineal desde la base de datos
    EN: Main function to run linear analysis from the database
    JA: DBから線形解析を実行するメイン関数"""
    try:
        # Obtener datos de la base de datos
        if filters:
            # ES: Aplicar filtros (implementar según la estructura de la BD)
            # EN: Apply filters (implement per DB structure)
            # JA: フィルタを適用（DB構造に応じて実装）
            query = "SELECT * FROM main_results WHERE 1=1"
            params = []
            
            for field, value in filters.items():
                if value and value != "":
                    if isinstance(value, tuple):  # Rango de valores
                        if value[0] and value[1]:
                            query += f" AND {field} BETWEEN ? AND ?"
                            params.extend([value[0], value[1]])
                    elif field in ['A13', 'A11', 'A21', 'A32']:  # Campos de cepillos
                        # ES: Filtrar por cepillo específico = 1 | EN: Filter by specific brush = 1 | JA: 特定ブラシ＝1でフィルタ
                        query += f" AND {field} = ?"
                        params.append(value)
                    else:  # Valor único
                        query += f" AND {field} = ?"
                        params.append(value)
            
            cursor = db_manager.conn.cursor()
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            data = cursor.fetchall()
            
            if not data:
                return {'success': False, 'error': 'No se encontraron datos con los filtros especificados'}
            
            df = pd.DataFrame(data, columns=columns)
            
        else:
            # Sin filtros, obtener todos los datos
            try:
                # Obtener datos usando fetch_all
                data = db_manager.fetch_all('main_results')
                if not data:
                    return {'success': False, 'error': 'La tabla main_results está vacía'}
                
                # Obtener nombres de columnas
                cursor = db_manager.conn.cursor()
                cursor.execute("PRAGMA table_info(main_results)")
                columns_info = cursor.fetchall()
                column_names = [col[1] for col in columns_info]
                
                df = pd.DataFrame(data, columns=column_names)
                
            except Exception as e:
                print(f"⚠️ DBからデータ取得中にエラー: {e}")
                return {'success': False, 'error': f'Error accediendo a la base de datos: {str(e)}'}
        
        print(f"📊 取得データ: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # ES: Crear y ejecutar pipeline de análisis | EN: Create and run analysis pipeline | JA: 解析パイプラインを作成・実行
        pipeline = LinearAnalysisPipeline()
        results = pipeline.run_analysis(df)
        
        return results
        
    except Exception as e:
        print(f"❌ 線形解析の実行中にエラー: {e}")
        return {
            'success': False,
            'error': str(e)
        }
