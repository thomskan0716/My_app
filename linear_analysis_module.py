



#!/usr/bin/env python
# coding: utf-8

"""
Módulo de Análisis Lineal para 0.00sec
Basado en 線形モデル_回帰分離混合_Ver2_noA11A21A32.py
Adaptado para trabajar con la base de datos del proyecto
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
    """Configuración del análisis lineal"""
    
    # Columnas objetivo (variables dependientes)
    TARGET_COLUMNS = ['バリ除去', '摩耗量', '上面ダレ量', '側面ダレ量']
    
    # Tipos de tarea para cada objetivo
    TARGET_TYPES = {
        'バリ除去': 'classification',
        '摩耗量': 'regression',
        '上面ダレ量': 'regression', 
        '側面ダレ量': 'regression'
    }
    
    # Columnas de características (variables independientes)
    # Mapeo de nombres de la BD a nombres del análisis
    FEATURE_COLUMNS = [
        '送り速度', 'UPカット', '切込量', 
        '突出量', '載せ率', '回転速度', 'パス数'  # Corregido: BD usa '突出量'
    ]
    
    # Mapeo de nombres de la BD a nombres del análisis
    # Nota: La BD usa '突出量' pero el análisis espera '突出量' (sin し)
    DB_TO_ANALYSIS_MAPPING = {
        '送り速度': '送り速度',
        'UPカット': 'UPカット', 
        '切込量': '切込量',
        '突出量': '突出量',  # Corregido: BD usa '突出量'
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
    """Pipeline de análisis lineal simplificado"""
    
    def __init__(self, output_dir: str = "output_analysis"):
        """Inicializar el pipeline"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.transformation_info = {}
        
        # Configurar matplotlib para japonés
        self._setup_japanese_font()
    
    def _setup_japanese_font(self):
        """Configurar fuente japonesa para matplotlib"""
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
            print(f"⚠️ No se pudo configurar fuente japonesa: {e}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preparar datos para el análisis"""
        print("🔧 Preparando datos para análisis...")
        
        # Mapear nombres de columnas de la BD a nombres del análisis
        column_mapping = {}
        for db_col, analysis_col in LinearAnalysisConfig.DB_TO_ANALYSIS_MAPPING.items():
            if db_col in df.columns:
                column_mapping[db_col] = analysis_col
        
        # Crear DataFrame con nombres mapeados
        df_mapped = df.rename(columns=column_mapping)
        
        # Seleccionar solo las columnas necesarias
        available_features = [col for col in LinearAnalysisConfig.FEATURE_COLUMNS 
                            if col in df_mapped.columns]
        available_targets = [col for col in LinearAnalysisConfig.TARGET_COLUMNS 
                           if col in df_mapped.columns]
        
        print(f"🔧 Características disponibles: {available_features}")
        print(f"🔧 Objetivos disponibles: {available_targets}")
        
        # Crear X (características) e y (objetivos)
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
        print("🔧 Entrenando modelos...")
        
        for target_col in y.columns:
            if target_col not in LinearAnalysisConfig.TARGET_TYPES:
                continue
                
            task_type = LinearAnalysisConfig.TARGET_TYPES[target_col]
            print(f"🔧 Entrenando modelo para {target_col} ({task_type})")
            
            try:
                # Obtener datos válidos para este objetivo
                valid_mask = ~y[target_col].isnull()
                X_valid = X[valid_mask]
                y_valid = y[target_col][valid_mask]
                
                if len(X_valid) < 10:
                    print(f"⚠️ Insuficientes datos para {target_col}: {len(X_valid)} muestras")
                    continue
                
                if task_type == 'regression':
                    model_info = self._train_regression_model(X_valid, y_valid, target_col)
                else:
                    model_info = self._train_classification_model(X_valid, y_valid, target_col)
                
                self.models[target_col] = model_info
                
            except Exception as e:
                print(f"❌ Error entrenando modelo para {target_col}: {e}")
                self.models[target_col] = {'error': str(e)}
    
    def _train_regression_model(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict:
        """Entrenar modelo de regresión"""
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=LinearAnalysisConfig.RANDOM_STATE),
            'Lasso': Lasso(random_state=LinearAnalysisConfig.RANDOM_STATE, max_iter=2000)
        }
        
        best_model_name = None
        best_score = -float('inf')
        best_model = None
        
        # Validación cruzada simple
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
                print(f"⚠️ Error con {name}: {e}")
        
        if best_model is None:
            best_model = LinearRegression()
            best_model_name = 'LinearRegression'
        
        # Entrenar el mejor modelo
        best_model.fit(X, y)
        y_pred = best_model.predict(X)
        
        # Métricas finales
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Guardar modelo
        model_path = self.output_dir / f'model_{target_name}.pkl'
        model_data = {
            'model': best_model,
            'feature_names': X.columns.tolist(),
            'target_name': target_name,
            'model_name': best_model_name
        }
        joblib.dump(model_data, model_path)
        
        # Crear gráfico
        self._plot_regression_results(y, y_pred, target_name)
        
        return {
            'model': best_model,
            'model_name': best_model_name,
            'model_path': str(model_path),
            'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
            'task_type': 'regression'
        }
    
    def _train_classification_model(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict:
        """Entrenar modelo de clasificación"""
        # Verificar que hay suficientes muestras por clase
        class_counts = y.value_counts()
        if len(class_counts) < 2 or class_counts.min() < 5:
            return {'error': 'insufficient_samples'}
        
        # Codificar etiquetas
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Entrenar modelo
        model = LogisticRegression(random_state=LinearAnalysisConfig.RANDOM_STATE, max_iter=2000)
        model.fit(X, y_encoded)
        
        # Predicciones
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # Métricas
        accuracy = accuracy_score(y_encoded, y_pred)
        f1 = f1_score(y_encoded, y_pred, average='weighted')
        
        # Guardar modelo
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
        """Crear gráfico de resultados de regresión"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico 1: Predicción vs Real
            ax1.scatter(y_true, y_pred, alpha=0.6)
            ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax1.set_xlabel('Valor Real')
            ax1.set_ylabel('Predicción')
            ax1.set_title(f'{target_name}: Predicción vs Real')
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Residuales
            residuals = y_true - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6)
            ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('Predicción')
            ax2.set_ylabel('Residuales')
            ax2.set_title(f'{target_name}: Análisis de Residuales')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar gráfico
            plot_path = self.output_dir / f'regression_{target_name}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Gráfico guardado: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Error creando gráfico para {target_name}: {e}")
    
    def save_results(self):
        """Guardar resultados del análisis"""
        print("🔧 Guardando resultados...")
        
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
        
        # Guardar como Excel
        results_df = pd.DataFrame(results_summary)
        results_path = self.output_dir / 'analysis_results.xlsx'
        results_df.to_excel(results_path, index=False)
        print(f"✅ Resultados guardados: {results_path}")
        
        # Guardar como JSON
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
        print(f"✅ Resultados JSON guardados: {json_path}")
    
    def run_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Ejecutar análisis completo"""
        print("🚀 Iniciando análisis lineal...")
        
        try:
            # Preparar datos
            X, y = self.prepare_data(df)
            
            if X.empty or y.empty:
                raise ValueError("No hay datos válidos para el análisis")
            
            print(f"✅ Datos preparados: {X.shape[0]} muestras, {X.shape[1]} características")
            
            # Entrenar modelos
            self.train_models(X, y)
            
            # Guardar resultados
            self.save_results()
            
            print("✅ Análisis lineal completado exitosamente")
            
            return {
                'success': True,
                'models': self.models,
                'output_directory': str(self.output_dir),
                'data_shape': X.shape
            }
            
        except Exception as e:
            print(f"❌ Error en análisis lineal: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def run_linear_analysis_from_db(db_manager, filters: Dict = None) -> Dict[str, Any]:
    """Función principal para ejecutar análisis lineal desde la base de datos"""
    try:
        # Obtener datos de la base de datos
        if filters:
            # Aplicar filtros (implementar según la estructura de la BD)
            query = "SELECT * FROM main_results WHERE 1=1"
            params = []
            
            for field, value in filters.items():
                if value and value != "":
                    if isinstance(value, tuple):  # Rango de valores
                        if value[0] and value[1]:
                            query += f" AND {field} BETWEEN ? AND ?"
                            params.extend([value[0], value[1]])
                    elif field in ['A13', 'A11', 'A21', 'A32']:  # Campos de cepillos
                        # Filtrar por cepillo específico = 1
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
                print(f"⚠️ Error obteniendo datos de la BD: {e}")
                return {'success': False, 'error': f'Error accediendo a la base de datos: {str(e)}'}
        
        print(f"📊 Datos obtenidos: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Crear y ejecutar pipeline de análisis
        pipeline = LinearAnalysisPipeline()
        results = pipeline.run_analysis(df)
        
        return results
        
    except Exception as e:
        print(f"❌ Error ejecutando análisis lineal: {e}")
        return {
            'success': False,
            'error': str(e)
        }
