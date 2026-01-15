import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

class Dsaitekika:
    @staticmethod
    def calculate_d_criterion(X_selected):
        try:
            q, r = np.linalg.qr(X_selected)
            det = np.abs(np.prod(np.diag(r)))
            return np.log(det) if det > 0 else float('-inf')
        except:
            return float('-inf')

    @staticmethod
    def calculate_i_criterion(X_selected):
        try:
            distances = cdist(X_selected, X_selected)
            np.fill_diagonal(distances, np.inf)
            return np.min(distances)
        except:
            return float('-inf')

    @staticmethod
    def select_optimal_design(X, num_points, output_prefix, num_trials=1000):
        best_combined_criterion = float('-inf')
        best_indices = None
        all_d_criteria = []
        all_i_criteria = []
        all_combined = []

        for trial in range(num_trials):
            indices = np.random.choice(len(X), num_points, replace=False)
            X_selected = X[indices]
            d_criterion = Dsaitekika.calculate_d_criterion(X_selected)
            i_criterion = Dsaitekika.calculate_i_criterion(X_selected)
            combined = d_criterion + i_criterion
            all_d_criteria.append(d_criterion)
            all_i_criteria.append(i_criterion)
            all_combined.append(combined)

            if combined > best_combined_criterion:
                best_combined_criterion = combined
                best_indices = indices

        # Guardar gráficos
        plt.figure(figsize=(10, 6))
        plt.hist(all_d_criteria, bins=50, alpha=0.7, label='D-criterion')
        plt.hist(all_i_criteria, bins=50, alpha=0.7, label='I-criterion')
        plt.hist(all_combined, bins=50, alpha=0.7, label='D+I Combined')
        plt.legend()
        plt.title('Distribution of Criteria Values')
        plt.xlabel('Criterion Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_criteria_distribution.png')
        plt.close()

        return best_indices, all_d_criteria, all_i_criteria, all_combined

    @staticmethod
    def create_visualization(df_filtered, X, optimal_indices, output_prefix):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X_scaled)

        result_values = np.zeros(len(df_filtered))

        # PCA Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=result_values, cmap='viridis', alpha=0.6, s=100)
        plt.scatter(X_pca[optimal_indices, 0], X_pca[optimal_indices, 1], color='red', marker='x', s=200)
        for idx, (x, y) in enumerate(X_pca[optimal_indices]):
            plt.text(x, y, str(df_filtered.index[optimal_indices][idx] + 1), fontsize=12, color='black', ha='center', va='center', weight='bold')
        plt.title('PCA Visualization')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_pca.png', dpi=300)
        plt.close()

        # UMAP Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=result_values, cmap='viridis', alpha=0.6, s=100)
        plt.scatter(X_umap[optimal_indices, 0], X_umap[optimal_indices, 1], color='red', marker='x', s=200)
        for idx, (x, y) in enumerate(X_umap[optimal_indices]):
            plt.text(x, y, str(df_filtered.index[optimal_indices][idx] + 1), fontsize=12, color='black', ha='center', va='center', weight='bold')
        plt.title('UMAP Visualization')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_umap.png', dpi=300)
        plt.close()

        # PCA Features
        feature_importance = pd.DataFrame(
            pca.components_.T, columns=['PC1', 'PC2'],
            index=[f'Feature {i + 1}' for i in range(X.shape[1])]
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(feature_importance, cmap='RdBu', center=0, annot=True)
        plt.title('PCA Feature Contributions')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_pca_features.png', dpi=300)
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
        design_df = df[design_cols].copy()
        X = design_df.values

        optimal_indices, d_vals, i_vals, combined_vals = Dsaitekika.select_optimal_design(X, num_points, output_prefix)

        selected_data = df.iloc[optimal_indices].copy()
        # Añadir columna No. con el número de muestra original (1-indexado)
        selected_data['No.'] = (df.index[optimal_indices] + 1).astype(int)
        # Reordenar para que No. sea la primera columna
        cols = ['No.'] + [c for c in selected_data.columns if c != 'No.']
        selected_data = selected_data[cols]
        
        # Calcular el mejor criterio D
        best_d_criterion = d_vals[np.argmax([d + i for d, i in zip(d_vals, i_vals)])]
        
        # Definir el orden y nombres de las columnas requeridas
        # Nota: la columna de dirección puede ser "UPカット" o "回転方向" según el archivo de entrada.
        required_columns = ['No.', 'A13', 'A11', 'A21', 'A32',
                            '回転速度', '送り速度', dir_col, '切込量', '突出量', '載せ率', 'パス数',
                            'D基準値', '上面ダレ', '側面ダレ', '摩耗量', '面粗度(Ra)前', '面粗度(Ra)後', '実験日']
        
        # Crear las columnas que falten
        for col in required_columns:
            if col not in selected_data.columns and col != 'D基準値':
                selected_data[col] = ''
        
        # Insertar la columna D基準値 con el valor igual para todas las filas
        selected_data['D基準値'] = best_d_criterion
        
        # Reordenar las columnas
        selected_data = selected_data[required_columns]
        
        viz_results = Dsaitekika.create_visualization(design_df, X, optimal_indices, output_prefix)

        selected_data.to_excel(output_excel_path, index=False)

        results = {
            'num_selected': len(optimal_indices),
            'pca_variance_ratio': viz_results['pca_variance_ratio'],
            'feature_importance': viz_results['feature_importance'],
            'selected_dataframe': selected_data,
            'best_d_criterion': best_d_criterion
        }
        print('DEBUG results en dsaitekika.py antes de return:', results)
        return results
