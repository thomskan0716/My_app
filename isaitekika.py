import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

class ISaitekika:
    @staticmethod
    def calculate_i_criterion(X_selected, X_all):
        # I基準値はGUI/Excel出力では常に空欄にする要件のため、
        # ここでは計算しない（互換のため関数自体は残す）。
        return ''

    @staticmethod
    def select_optimal_design(X, num_points, num_trials=1000):
        # I基準値は計算しない。選択ロジックは従来通りランダム探索で良いが、
        # best_criterion は空欄を返す。
        best_criterion = ''
        best_indices = None
        all_criteria = []

        for trial in range(num_trials):
            indices = np.random.choice(len(X), num_points, replace=False)
            X_selected = X[indices]
            # ここでは分布保存用に距離指標を計算することも可能だが、
            # 要件に合わせて I基準値は出力しない。
            criterion = None
            all_criteria.append(0)
            if best_indices is None:
                best_indices = indices

        # ES: Guardar distribución | EN: Save distribution | JA: 分布を保存
        plt.figure(figsize=(10, 6))
        plt.hist(all_criteria, bins=50)
        plt.title('Distribution of I-criterion Values')
        plt.xlabel('I-criterion Value')
        plt.ylabel('Frequency')
        plt.savefig('i_criterion_distribution.png')
        plt.close()

        return best_indices, best_criterion

    @staticmethod
    def create_visualization(df_filtered, X, optimal_indices, output_prefix):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X_scaled)

        result_values = np.zeros(len(df_filtered))

        # PCA plot
        plt.figure(figsize=(10, 8))
        scatter_pca = plt.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=result_values, cmap='viridis', alpha=0.6, s=100
        )
        plt.scatter(X_pca[optimal_indices, 0], X_pca[optimal_indices, 1],
                    color='blue', marker='x', s=200, label='Selected Points')
        for idx, (x, y) in enumerate(X_pca[optimal_indices]):
            plt.text(x, y, str(df_filtered.index[optimal_indices][idx] + 1), fontsize=12, color='black',
                     ha='center', va='center', weight='bold')
        plt.title(f'PCA Visualization\n(Explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}, '
                  f'{pca.explained_variance_ratio_[1]:.3f})')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.colorbar(scatter_pca, label='Result Value')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_pca.png', dpi=300, bbox_inches='tight')
        plt.close()

        # UMAP plot
        plt.figure(figsize=(10, 8))
        scatter_umap = plt.scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=result_values, cmap='viridis', alpha=0.6, s=100
        )
        plt.scatter(X_umap[optimal_indices, 0], X_umap[optimal_indices, 1],
                    color='blue', marker='x', s=200, label='Selected Points')
        for idx, (x, y) in enumerate(X_umap[optimal_indices]):
            plt.text(x, y, str(df_filtered.index[optimal_indices][idx] + 1), fontsize=12, color='black',
                     ha='center', va='center', weight='bold')
        plt.title('UMAP Visualization')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.legend()
        plt.colorbar(scatter_umap, label='Result Value')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_umap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # PCA contributions
        plt.figure(figsize=(10, 6))
        feature_importance = pd.DataFrame(
            pca.components_.T, columns=['PC1', 'PC2'],
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

    @staticmethod
    def run(input_excel_path, output_excel_path, output_prefix, num_points=15):
        ext = os.path.splitext(str(input_excel_path))[1].lower()
        if ext == ".csv":
            df = pd.read_csv(input_excel_path, encoding="utf-8-sig")
        else:
            df = pd.read_excel(input_excel_path)
        # Variables de diseño por nombre (compatibilidad con nuevo formato con A13/A11/A21/A32 al inicio)
        dir_col = "UPカット" if "UPカット" in df.columns else ("回転方向" if "回転方向" in df.columns else None)
        if dir_col is None:
            raise ValueError("❌ Falta columna de dirección: 'UPカット' o '回転方向'")
        design_cols = ["回転速度", "送り速度", dir_col, "切込量", "突出量", "載せ率", "パス数"]
        missing = [c for c in design_cols if c not in df.columns]
        if missing:
            raise ValueError(f"❌ Faltan columnas de diseño: {missing}")
        df_filtered = df[design_cols].copy()
        X = df_filtered.values

        optimal_indices, best_criterion = ISaitekika.select_optimal_design(X, num_points)

        selected_data = df.iloc[optimal_indices].copy()
        # Añadir columna No. con el número de muestra original (1-indexado)
        selected_data['No.'] = (df.index[optimal_indices] + 1).astype(int)
        # Reordenar para que No. sea la primera columna
        cols = ['No.'] + [c for c in selected_data.columns if c != 'No.']
        selected_data = selected_data[cols]

        # Insertar A13/A11/A21/A32 justo después de No. (según sample_combinations)
        a_cols = ['A13', 'A11', 'A21', 'A32']
        for c in a_cols:
            if c not in selected_data.columns:
                selected_data[c] = ''
        ordered_cols = ['No.'] + a_cols + [c for c in selected_data.columns if c not in (['No.'] + a_cols)]
        selected_data = selected_data[ordered_cols]
        
        viz_results = ISaitekika.create_visualization(df_filtered, X, optimal_indices, output_prefix)

        # ISaitekika: I基準値は常に空欄（Excel出力側でも上書き）
        if 'I基準値' not in selected_data.columns:
            selected_data['I基準値'] = ''
        else:
            selected_data['I基準値'] = ''
        selected_data.to_excel(output_excel_path, index=False)

        results = {
            'num_selected': len(optimal_indices),
            'best_criterion': best_criterion,
            'pca_variance_ratio': viz_results['pca_variance_ratio'],
            'feature_importance': viz_results['feature_importance'],
            'selected_dataframe': selected_data,
            'best_i_criterion': ''
        }
        print('DEBUG results en isaitekika.py antes de return:', results)
        return results
