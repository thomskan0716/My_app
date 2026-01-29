# 修正３ ランダムに組み合わせたサンプリングを行うように修正

import pandas as pd
import numpy as np
from tkinter import Tk, filedialog, simpledialog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def calculate_d_criterion(X_selected):
    """D最適化基準値の計算"""
    try:
        q, r = np.linalg.qr(X_selected)
        det = np.abs(np.prod(np.diag(r)))
        return np.log(det) if det > 0 else float('-inf')
    except:
        return float('-inf')


def select_optimal_design(X, num_points, num_trials=10000000):
    """ランダムな組み合わせからD最適な実験点を選択"""
    best_criterion = float('-inf')
    best_indices = None
    all_criteria = []

    print(f"\n{num_trials}回の組み合わせを評価中...")

    for trial in range(num_trials):
        # ランダムな実験点の選択
        indices = np.random.choice(len(X), num_points, replace=False)
        X_selected = X[indices]

        # D最適化基準値の計算
        criterion = calculate_d_criterion(X_selected)
        all_criteria.append(criterion)

        # より良い解が見つかった場合は更新
        if criterion > best_criterion:
            best_criterion = criterion
            best_indices = indices
            print(f"Trial {trial + 1}: 新しい最良解を発見 (D基準値: {best_criterion:.3f})")

    # D基準値の分布をプロット
    plt.figure(figsize=(10, 6))
    plt.hist(all_criteria, bins=50)
    plt.title('Distribution of D-criterion Values')
    plt.xlabel('D-criterion Value')
    plt.ylabel('Frequency')
    plt.savefig('d_criterion_distribution.png')
    plt.close()

    return best_indices, best_criterion


def create_visualization(df_filtered, X, optimal_indices, output_prefix):
    """次元削減による可視化とプロット作成"""
    print("\n次元削減による可視化を作成中...")

    # データの標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCAの実行
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # UMAPの実行
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # プロット用の結果値（H列）を取得
    result_values = df_filtered.iloc[:, 7].values

    # プロットの作成
    fig, axs = plt.subplots(2, 1, figsize=(12, 16))
    plt.subplots_adjust(hspace=0.3)

    # PCAプロット
    scatter_pca = axs[0].scatter(X_pca[:, 0], X_pca[:, 1],
                                 c=result_values, cmap='viridis',
                                 alpha=0.6, s=100)
    axs[0].scatter(X_pca[optimal_indices, 0], X_pca[optimal_indices, 1],
                   color='red', marker='x', s=200, label='Selected Points')

    axs[0].set_title(f'PCA Visualization\n'
                     f'(Explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}, '
                     f'{pca.explained_variance_ratio_[1]:.3f})')
    axs[0].set_xlabel('First Principal Component')
    axs[0].set_ylabel('Second Principal Component')
    axs[0].legend()
    fig.colorbar(scatter_pca, ax=axs[0], label='Result Value')

    # UMAPプロット
    scatter_umap = axs[1].scatter(X_umap[:, 0], X_umap[:, 1],
                                  c=result_values, cmap='viridis',
                                  alpha=0.6, s=100)
    axs[1].scatter(X_umap[optimal_indices, 0], X_umap[optimal_indices, 1],
                   color='red', marker='x', s=200, label='Selected Points')

    axs[1].set_title('UMAP Visualization')
    axs[1].set_xlabel('UMAP1')
    axs[1].set_ylabel('UMAP2')
    axs[1].legend()
    fig.colorbar(scatter_umap, ax=axs[1], label='Result Value')

    plt.suptitle('Dimension Reduction Visualization of Experimental Points',
                 fontsize=14, y=0.95)

    # プロットの保存
    plt.savefig(f'{output_prefix}_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    # PCA寄与度の可視化
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=[f'Feature {i + 1}' for i in range(X.shape[1])]
    )

    sns.heatmap(feature_importance, cmap='RdBu', center=0, annot=True)
    plt.title('PCA Feature Contributions')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_pca_features.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'pca_variance_ratio': pca.explained_variance_ratio_,
        'feature_importance': feature_importance
    }


def main():
    # 実験点数の設定
    root = Tk()
    root.call('wm', 'attributes', '.', '-topmost', True)
    num_points = simpledialog.askinteger(
        title="実験点数の設定",
        prompt="選択する実験点の数を入力してください（推奨：15-20点）:",
        initialvalue=15
    )
    if not num_points:
        print("実験点数が設定されませんでした。")
        root.destroy()
        return

    # ファイル選択
    file_path = filedialog.askopenfilename(
        parent=root,
        title="Excelファイルを選択してください",
        filetypes=[("Excel Files", "*.xlsx *.xls")]
    )
    root.destroy()

    if not file_path:
        print("ファイルが選択されませんでした。")
        return

    # データ読み込みと前処理
    print("\nデータ読み込み中...")
    df = pd.read_excel(file_path)
    df_filtered = df.iloc[:, :6].copy()
    X = df_filtered.values

    # df_filtered = df[df.iloc[:, 7] >= 0.5]
    # X = df_filtered.iloc[:, :7].values

    print(f"\nフィルタリング後のデータ数: {len(df_filtered)}行")

    # D最適な実験点の選択
    optimal_indices, best_criterion = select_optimal_design(
        X,
        num_points=num_points,
        num_trials=1000
    )

    # 結果のデータフレーム作成
    d_optimal_data = df_filtered.iloc[optimal_indices].copy()
    d_optimal_data['D基準値'] = best_criterion

    # 結果の保存
    output_path = file_path.replace(".xlsx", f"_D_optimal_{num_points}points")
    d_optimal_data.to_excel(f"{output_path}.xlsx", index=False)

    # 可視化の実行
    viz_results = create_visualization(df_filtered, X, optimal_indices, output_path)

    print("\n=== 実験計画の結果 ===")
    print(f"選択された実験点数: {len(optimal_indices)}")
    print(f"最終的なD基準値: {best_criterion:.3f}")

    print("\n=== 次元削減の結果 ===")
    print(f"PCA説明寄与率:")
    print(f"- 第1主成分: {viz_results['pca_variance_ratio'][0]:.3f}")
    print(f"- 第2主成分: {viz_results['pca_variance_ratio'][1]:.3f}")

    print(f"\n結果を保存しました:")
    print(f"- D最適実験計画: {output_path}.xlsx")
    print(f"- 次元削減プロット: {output_path}_visualization.png")
    print(f"- PCA特徴量寄与度: {output_path}_pca_features.png")
    print(f"- D基準値分布: d_criterion_distribution.png")


if __name__ == "__main__":
    main()