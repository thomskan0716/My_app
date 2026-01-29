"""
完全版SHAP分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ES: SHAP se importa de forma lazy (solo cuando se necesita) para evitar conflictos
# EN: SHAP is imported lazily (only when needed) to avoid conflicts
# JP: 競合を避けるためSHAPは遅延インポート（必要時のみ）する
# ES: con matplotlib después de crear muchos gráficos
# EN: with matplotlib after creating many plots
# JP: 多数のグラフ生成後にmatplotlibと競合するのを防ぐため
SHAP_AVAILABLE = None
_shap_module = None

def _get_shap():
    """Lazy import de SHAP - solo se importa cuando se necesita realmente"""
    global SHAP_AVAILABLE, _shap_module
    if SHAP_AVAILABLE is None:
        try:
            import shap
            _shap_module = shap
            SHAP_AVAILABLE = True
        except ImportError:
            _shap_module = None
            SHAP_AVAILABLE = False
    return _shap_module if SHAP_AVAILABLE else None

class CompleteSHAPAnalyzer:
    """完全版SHAP分析クラス"""
    
    def __init__(self, config):
        self.config = config
        self.shap_values = None
        self.explainer = None
        
    def analyze(self, model, X, y, feature_names, target_name, model_type, output_folder):
        """完全なSHAP分析実行"""
        
        shap = _get_shap()
        if shap is None:
            print("⚠ SHAP not available")
            return None
            
        print(f"\n{'='*60}")
        print(f"完全版SHAP分析: {target_name}")
        print(f"Model type: {model_type}")
        print(f"{'='*60}")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # サンプリング
        n_samples = min(self.config.SHAP_MAX_SAMPLES, len(X))
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_shap = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
            y_shap = y[indices] if hasattr(y, '__getitem__') else y
        else:
            X_shap = X
            y_shap = y
            
        print(f"SHAP分析サンプル数: {n_samples}")
        
        # Explainer選択
        self.explainer = self._create_explainer(model, X_shap, model_type)
        
        # SHAP値計算
        print("SHAP値計算中...")
        self.shap_values = self._calculate_shap_values(X_shap)
        
        if self.shap_values is None:
            print("⚠ SHAP値計算失敗")
            return None
            
        # 各種プロット生成
        self._generate_summary_plot(X_shap, feature_names, target_name, output_folder)
        self._generate_importance_plot(feature_names, target_name, output_folder)
        self._generate_waterfall_plots(X_shap, feature_names, target_name, output_folder)
        self._generate_dependence_plots(X_shap, feature_names, target_name, output_folder)
        self._generate_force_plots(X_shap, feature_names, target_name, output_folder)
        self._generate_interaction_plots(X_shap, feature_names, target_name, output_folder)
        
        # 特徴量重要度保存
        self._save_feature_importance(feature_names, output_folder)
        
        print(f"\n✅ SHAP分析完了: {output_folder}")
        
        return self.shap_values
    
    def _create_explainer(self, model, X_shap, model_type):
        """適切なExplainerを作成"""
        shap = _get_shap()
        if shap is None:
            return None
        
        if model_type == 'tree':
            print("TreeExplainer使用")
            return shap.TreeExplainer(model)
            
        elif model_type == 'linear':
            print("LinearExplainer使用")
            return shap.LinearExplainer(model, X_shap)
            
        else:
            print("KernelExplainer使用")
            background = shap.sample(X_shap, min(100, len(X_shap)))
            return shap.KernelExplainer(model.predict, background)
    
    def _calculate_shap_values(self, X_shap):
        """SHAP値計算"""
        shap = _get_shap()
        if shap is None:
            return None
        try:
            if isinstance(self.explainer, shap.TreeExplainer):
                return self.explainer.shap_values(X_shap)
            elif isinstance(self.explainer, shap.LinearExplainer):
                return self.explainer.shap_values(X_shap)
            else:
                # KernelExplainer（計算量削減）
                return self.explainer.shap_values(X_shap, nsamples=100)
        except Exception as e:
            print(f"SHAP値計算エラー: {e}")
            return None
    
    def _generate_summary_plot(self, X_shap, feature_names, target_name, output_folder):
        """Summary Plot生成"""
        shap = _get_shap()
        if shap is None:
            return
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(self.shap_values, X_shap, 
                            feature_names=feature_names, 
                            show=False)
            plt.title(f'{target_name} - SHAP Summary Plot')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'summary_plot.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Summary plot生成")
        except Exception as e:
            print(f"× Summary plotエラー: {e}")
    
    def _generate_importance_plot(self, feature_names, target_name, output_folder):
        """重要度バープロット生成"""
        shap = _get_shap()
        if shap is None:
            return
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, feature_names=feature_names,
                            plot_type="bar", show=False)
            plt.title(f'{target_name} - Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'importance_bar.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Importance bar plot生成")
        except Exception as e:
            print(f"× Importance plotエラー: {e}")
    
    def _generate_waterfall_plots(self, X_shap, feature_names, target_name, output_folder):
        """Waterfall Plots生成"""
        shap = _get_shap()
        if shap is None:
            return
        try:
            # Explanationオブジェクト作成
            shap_explanation = shap.Explanation(
                values=self.shap_values,
                base_values=np.full(self.shap_values.shape[0], 
                              self.explainer.expected_value if hasattr(self.explainer, 'expected_value') 
                              else np.mean(self.shap_values)),
                data=X_shap,
                feature_names=feature_names
            )
            
            # 代表的なサンプル選択
            n_samples = min(5, len(X_shap))
            indices = [0, len(X_shap)//4, len(X_shap)//2, 3*len(X_shap)//4, len(X_shap)-1]
            indices = [i for i in indices if i < len(X_shap)][:n_samples]
            
            for idx in indices:
                plt.figure(figsize=(12, 6))
                shap.waterfall_plot(shap_explanation[idx], show=False)
                plt.title(f'{target_name} - Waterfall Plot (Sample {idx})')
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f'waterfall_sample_{idx}.png'),
                          dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"✓ Waterfall plots生成 ({n_samples}個)")
        except Exception as e:
            print(f"× Waterfall plotsエラー: {e}")
    
    def _generate_dependence_plots(self, X_shap, feature_names, target_name, output_folder):
        """Dependence Plots生成"""
        shap = _get_shap()
        if shap is None:
            return
        try:
            # 重要度上位6特徴量
            importance = np.abs(self.shap_values).mean(0)
            top_features_idx = np.argsort(importance)[-6:]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, feat_idx in enumerate(top_features_idx):
                ax = axes[i]
                shap.dependence_plot(feat_idx, self.shap_values, X_shap,
                                    feature_names=feature_names,
                                    ax=ax, show=False)
            
            plt.suptitle(f'{target_name} - SHAP Dependence Plots', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'dependence_plots.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Dependence plots生成")
        except Exception as e:
            print(f"× Dependence plotsエラー: {e}")
    
    def _generate_force_plots(self, X_shap, feature_names, target_name, output_folder):
        """Force Plots生成"""
        shap = _get_shap()
        if shap is None:
            return
        try:
            # 単一サンプルのforce plot
            expected_value = self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
            
            # 複数サンプルのforce plot
            force_plot = shap.force_plot(
                expected_value,
                self.shap_values[:10],
                X_shap[:10],
                feature_names=feature_names
            )
            
            # HTMLとして保存
            shap.save_html(os.path.join(output_folder, 'force_plot.html'), force_plot)
            print("✓ Force plot (HTML)生成")
        except Exception as e:
            print(f"× Force plotエラー: {e}")
    
    def _generate_interaction_plots(self, X_shap, feature_names, target_name, output_folder):
        """相互作用プロット生成"""
        try:
            # 化学成分ブラシとの相互作用
            brush_features = self.config.DISCRETE_FEATURES
            continuous_features = self.config.CONTINUOUS_FEATURES[:3]  # 上位3つ
            
            fig, axes = plt.subplots(len(continuous_features), len(brush_features), 
                                    figsize=(5*len(brush_features), 5*len(continuous_features)))
            
            if len(continuous_features) == 1:
                axes = axes.reshape(1, -1)
            
            for i, cont_feat in enumerate(continuous_features):
                for j, brush_feat in enumerate(brush_features):
                    ax = axes[i, j] if len(continuous_features) > 1 else axes[j]
                    
                    # 特徴量インデックス取得
                    cont_idx = feature_names.index(cont_feat) if cont_feat in feature_names else i
                    brush_idx = feature_names.index(brush_feat) if brush_feat in feature_names else j
                    
                    # 散布図
                    scatter = ax.scatter(X_shap[:, cont_idx] if isinstance(X_shap, np.ndarray) else X_shap.iloc[:, cont_idx],
                                       self.shap_values[:, cont_idx],
                                       c=X_shap[:, brush_idx] if isinstance(X_shap, np.ndarray) else X_shap.iloc[:, brush_idx],
                                       cmap='viridis', alpha=0.6)
                    
                    ax.set_xlabel(cont_feat)
                    ax.set_ylabel(f'SHAP value for {cont_feat}')
                    ax.set_title(f'{cont_feat} vs {brush_feat}')
                    plt.colorbar(scatter, ax=ax, label=brush_feat)
            
            plt.suptitle(f'{target_name} - Feature Interactions', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'interaction_plots.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Interaction plots生成")
        except Exception as e:
            print(f"× Interaction plotsエラー: {e}")
    
    def _save_feature_importance(self, feature_names, output_folder):
        """特徴量重要度をCSV保存"""
        try:
            importance = np.abs(self.shap_values).mean(0)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv(os.path.join(output_folder, 'feature_importance.csv'),
                               index=False)
            print("✓ 特徴量重要度CSV保存")
        except Exception as e:
            print(f"× CSV保存エラー: {e}")