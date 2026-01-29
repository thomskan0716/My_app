#!/usr/bin/env python
# coding: utf-8
"""
特徴量重要度可視化機能のテストスクリプト
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# パス設定（ml_modulesフォルダ内から実行する場合）
BASE = Path("../")  # 親ディレクトリを指す
MODULES_DIR = BASE / "ml_modules"
if MODULES_DIR.exists():
    if str(MODULES_DIR) not in sys.path:
        sys.path.insert(0, str(MODULES_DIR))

try:
    from feature_importance_plot import (
        extract_feature_importance,
        plot_feature_importance,
        create_feature_importance_analysis,
        compare_feature_importance_across_models
    )
    from config_cls import ConfigCLS
    print("✅ モジュールのインポートに成功")
except ImportError as e:
    print(f"❌ モジュールのインポートに失敗: {e}")
    sys.exit(1)


def create_sample_data():
    """テスト用のサンプルデータを作成"""
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    # 特徴量名
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # サンプルデータ生成
    X = np.random.randn(n_samples, n_features)
    # 最初の3つの特徴量を重要に設定
    y = (X[:, 0] + X[:, 1] * 2 + X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    return X, y, feature_names


def test_feature_importance_extraction():
    """特徴量重要度の抽出テスト"""
    print("\n=== 特徴量重要度抽出テスト ===")
    
    X, y, feature_names = create_sample_data()
    
    # RandomForestで学習
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    # 重要度抽出
    importance_series = extract_feature_importance(rf_model, feature_names)
    
    print(f"抽出された重要度:")
    print(importance_series.head())
    
    # 期待通り最初の3つの特徴量が重要になっているかチェック
    top_3_features = importance_series.head(3).index.tolist()
    expected_features = ['feature_0', 'feature_1', 'feature_2']
    
    if set(top_3_features) == set(expected_features):
        print("✅ 重要度抽出テスト成功")
    else:
        print(f"⚠️ 期待される特徴量と異なります: {top_3_features}")
    
    return importance_series


def test_feature_importance_plotting():
    """特徴量重要度の可視化テスト"""
    print("\n=== 特徴量重要度可視化テスト ===")
    
    X, y, feature_names = create_sample_data()
    
    # 複数のモデルで学習
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    for model_name, model in models.items():
        model.fit(X, y)
        importance_series = extract_feature_importance(model, feature_names)
        
        # 可視化（保存なし）
        print(f"{model_name}の特徴量重要度を可視化中...")
        plot_feature_importance(
            importance_series,
            save_path=None,  # 保存しない
            top_k=5,
            title=f"{model_name} - 特徴量重要度"
        )
    
    print("✅ 可視化テスト完了")


def test_feature_importance_analysis():
    """特徴量重要度分析のテスト"""
    print("\n=== 特徴量重要度分析テスト ===")
    
    X, y, feature_names = create_sample_data()
    
    # モデル学習
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 結果フォルダ作成
    result_folder = "test_results"
    os.makedirs(result_folder, exist_ok=True)
    
    # 分析実行
    analysis_result = create_feature_importance_analysis(
        model=model,
        feature_names=feature_names,
        result_folder=result_folder,
        model_name="test_model",
        top_k=5
    )
    
    print(f"分析結果:")
    print(f"  保存先: {analysis_result['save_path']}")
    print(f"  上位5個のカバレッジ: {analysis_result['top_k_coverage']:.1%}")
    
    # ファイルの存在確認
    if os.path.exists(analysis_result['save_path']):
        print("✅ PNGファイルが正常に保存されました")
    else:
        print("❌ PNGファイルの保存に失敗しました")
    
    return analysis_result


def test_model_comparison():
    """複数モデルの特徴量重要度比較テスト"""
    print("\n=== 複数モデル比較テスト ===")
    
    X, y, feature_names = create_sample_data()
    
    # 複数のモデルで学習
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    for model_name, model in models.items():
        model.fit(X, y)
    
    # 結果フォルダ作成
    result_folder = "test_results"
    os.makedirs(result_folder, exist_ok=True)
    
    # 比較分析実行
    compare_feature_importance_across_models(
        models_dict=models,
        feature_names=feature_names,
        result_folder=result_folder,
        top_k=5
    )
    
    print("✅ 複数モデル比較テスト完了")


def main():
    """メインテスト実行"""
    print("="*60)
    print("特徴量重要度可視化機能のテスト")
    print("="*60)
    
    try:
        # 各テストの実行
        test_feature_importance_extraction()
        test_feature_importance_plotting()
        test_feature_importance_analysis()
        test_model_comparison()
        
        print("\n" + "="*60)
        print("✅ すべてのテストが完了しました")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
