"""
特徴量重要度の可視化モジュール
- 学習済みモデルから特徴量重要度を抽出・可視化
- 複数のモデルタイプに対応
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from sklearn.base import BaseEstimator

from config_cls import ConfigCLS


def extract_feature_importance(model: BaseEstimator, feature_names: List[str]) -> pd.Series:
    """
    学習済みモデルから特徴量重要度を抽出
    
    Parameters:
    -----------
    model : BaseEstimator
        学習済みモデル
    feature_names : List[str]
        特徴量名のリスト
    
    Returns:
    --------
    importance_series : pd.Series
        特徴量重要度のSeries（重要度降順）
    """
    # モデルタイプに応じて重要度を取得
    if hasattr(model, 'feature_importances_'):
        # Tree系モデル（RandomForest, XGBoost, LightGBM等）
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # 線形モデル（LogisticRegression等）
        coef = model.coef_
        if coef.ndim == 2:  # (1, n_features) の形状の場合
            coef = coef[0]
        importance = np.abs(coef)  # 係数の絶対値を使用
    else:
        # 重要度が取得できない場合は一様重み
        importance = np.ones(len(feature_names))
        print("警告: モデルから特徴量重要度を取得できません。一様重みを使用します。")
    
    # Seriesに変換して重要度降順でソート
    importance_series = pd.Series(importance, index=feature_names)
    importance_series = importance_series.sort_values(ascending=False)
    
    return importance_series


def plot_feature_importance(
    importance_series: pd.Series,
    save_path: Optional[str] = None,
    top_k: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "特徴量重要度"
) -> None:
    """
    特徴量重要度を可視化
    
    Parameters:
    -----------
    importance_series : pd.Series
        特徴量重要度のSeries
    save_path : str, optional
        保存先パス（Noneの場合は表示のみ）
    top_k : int
        表示する上位特徴量数
    figsize : Tuple[int, int]
        図のサイズ
    title : str
        図のタイトル
    """
    # 日本語フォント設定
    plt.rcParams['font.family'] = ConfigCLS.PLOT_FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = ConfigCLS.PLOT_UNICODE_MINUS
    
    # 上位k個の特徴量を選択
    top_features = importance_series.head(top_k)
    
    # 図の作成
    fig, ax = plt.subplots(figsize=figsize)
    
    # 棒グラフの作成
    bars = ax.barh(range(len(top_features)), top_features.values, 
                   color='steelblue', alpha=0.7)
    
    # 軸の設定
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=10)
    ax.set_xlabel('重要度', fontsize=12)
    ax.set_title(f'{title}\n(上位{top_k}個の特徴量)', fontsize=14, fontweight='bold')
    
    # 重要度の値を棒の右端に表示
    for i, (bar, value) in enumerate(zip(bars, top_features.values)):
        ax.text(value + max(top_features.values) * 0.01, i, 
                f'{value:.3f}', va='center', fontsize=9)
    
    # グリッドの追加
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # レイアウト調整
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=ConfigCLS.PLOT_DPI, 
                   bbox_inches=ConfigCLS.PLOT_BBOX_INCHES)
        print(f"特徴量重要度図を保存: {save_path}")
    
    # 図を閉じてメモリを解放
    plt.close()


def create_feature_importance_analysis(
    model: BaseEstimator,
    feature_names: List[str],
    result_folder: str,
    model_name: str = "model",
    top_k: int = 20
) -> Dict[str, Any]:
    """
    特徴量重要度の完全な分析と可視化
    
    Parameters:
    -----------
    model : BaseEstimator
        学習済みモデル
    feature_names : List[str]
        特徴量名のリスト
    result_folder : str
        結果保存フォルダ
    model_name : str
        モデル名（ファイル名に使用）
    top_k : int
        表示する上位特徴量数
    
    Returns:
    --------
    analysis_result : Dict[str, Any]
        分析結果の辞書
    """
    print(f"\n=== 特徴量重要度分析 ({model_name}) ===")
    
    # 重要度の抽出
    importance_series = extract_feature_importance(model, feature_names)
    
    # 統計情報の計算
    total_importance = importance_series.sum()
    normalized_importance = importance_series / total_importance
    
    # 上位特徴量の情報
    top_features = importance_series.head(top_k)
    top_features_normalized = normalized_importance.head(top_k)
    
    print(f"総特徴量数: {len(importance_series)}")
    print(f"上位{top_k}個の特徴量が全体の{top_features_normalized.sum():.1%}を占める")
    print(f"\n上位{min(10, top_k)}個の特徴量:")
    for i, (feature, importance) in enumerate(top_features.head(10).items(), 1):
        print(f"  {i:2d}. {feature:15s}: {importance:.4f} ({normalized_importance[feature]:.1%})")
    
    # 可視化
    save_path = os.path.join(result_folder, f"feature_importance_{model_name}.png")
    plot_feature_importance(
        importance_series,
        save_path=save_path,
        top_k=top_k,
        title=f"特徴量重要度 ({model_name})"
    )
    
    # 分析結果の辞書
    analysis_result = {
        'importance_series': importance_series,
        'normalized_importance': normalized_importance,
        'top_features': top_features,
        'total_importance': total_importance,
        'top_k_coverage': top_features_normalized.sum(),
        'model_name': model_name,
        'save_path': save_path
    }
    
    return analysis_result


def compare_feature_importance_across_models(
    models_dict: Dict[str, BaseEstimator],
    feature_names: List[str],
    result_folder: str,
    top_k: int = 15
) -> None:
    """
    複数モデルの特徴量重要度を比較
    
    Parameters:
    -----------
    models_dict : Dict[str, BaseEstimator]
        モデル名とモデルオブジェクトの辞書
    feature_names : List[str]
        特徴量名のリスト
    result_folder : str
        結果保存フォルダ
    top_k : int
        比較する上位特徴量数
    """
    print(f"\n=== 複数モデルの特徴量重要度比較 ===")
    
    # 各モデルの重要度を取得
    importance_data = {}
    for model_name, model in models_dict.items():
        importance_series = extract_feature_importance(model, feature_names)
        importance_data[model_name] = importance_series.head(top_k)
    
    # 比較用のDataFrameを作成
    comparison_df = pd.DataFrame(importance_data)
    comparison_df = comparison_df.fillna(0)  # 欠損値を0で埋める
    
    # 可視化
    plt.rcParams['font.family'] = ConfigCLS.PLOT_FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = ConfigCLS.PLOT_UNICODE_MINUS
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # ヒートマップの作成
    sns.heatmap(comparison_df.T, 
                annot=True, 
                fmt='.3f', 
                cmap='YlOrRd',
                cbar_kws={'label': '重要度'},
                ax=ax)
    
    ax.set_title(f'複数モデルの特徴量重要度比較\n(上位{top_k}個の特徴量)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('特徴量', fontsize=12)
    ax.set_ylabel('モデル', fontsize=12)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(result_folder, "feature_importance_comparison.png")
    plt.savefig(save_path, dpi=ConfigCLS.PLOT_DPI, 
               bbox_inches=ConfigCLS.PLOT_BBOX_INCHES)
    print(f"特徴量重要度比較図を保存: {save_path}")
    
    # 図を閉じてメモリを解放
    plt.close()


if __name__ == "__main__":
    # テスト用のサンプルコード
    print("特徴量重要度可視化モジュールのテスト")
