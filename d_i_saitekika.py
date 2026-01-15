import pandas as pd
import numpy as np
import os
from itertools import product
from sklearn.preprocessing import StandardScaler
from scipy.linalg import qr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import seaborn as sns
import warnings

def generate_candidate_points(design_df):
    levels = []
    for _, row in design_df.iterrows():
        levels.append(np.arange(row["最小値"], row["最大値"] + row["刻み幅"], row["刻み幅"]))
    return np.array(list(product(*levels)))

def calculate_d_criterion_stable(X):
    try:
        q, r = qr(X, mode='economic')
        diag_r = np.diag(r)
        det = np.abs(np.prod(diag_r))
        log_det = np.log(det) if det > 1e-300 else -np.inf
        return log_det
    except Exception:
        return -np.inf

@staticmethod
def create_visualization(df_filtered, X, optimal_indices, output_prefix):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    result_values = np.zeros(len(df_filtered))  # Color neutro para todos

    # ========= PCA Plot =========
    plt.figure(figsize=(10, 8))
    scatter_pca = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=result_values, cmap='viridis', alpha=0.6, s=100
    )

    # Puntos seleccionados
    plt.scatter(X_pca[optimal_indices, 0], X_pca[optimal_indices, 1],
                color='red', marker='x', s=200, label='Selected Points')

    # ➕ Números del 1 al N sobre los puntos seleccionados
    for idx, (x, y) in enumerate(X_pca[optimal_indices]):
        plt.text(x, y, str(idx + 1), fontsize=12, color='black',
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

    # ========= UMAP Plot =========
    plt.figure(figsize=(10, 8))
    scatter_umap = plt.scatter(
        X_umap[:, 0], X_umap[:, 1],
        c=result_values, cmap='viridis', alpha=0.6, s=100
    )

    plt.scatter(X_umap[optimal_indices, 0], X_umap[optimal_indices, 1],
                color='red', marker='x', s=200, label='Selected Points')

    # ➕ Números sobre los puntos seleccionados
    for idx, (x, y) in enumerate(X_umap[optimal_indices]):
        plt.text(x, y, str(idx + 1), fontsize=12, color='black',
                 ha='center', va='center', weight='bold')

    plt.title('UMAP Visualization')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.colorbar(scatter_umap, label='Result Value')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_umap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========= PCA Feature Contributions =========
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

def run_d_i_saitekika(setting_file, output_folder="outputs", n_experiments=15):
    # Leer Excel
    design_df = pd.read_excel(setting_file, sheet_name="実験計画_説明変数設定")
    variable_names = design_df["説明変数名"].tolist()

    # Generar puntos candidatos
    candidate_points = generate_candidate_points(design_df)
    candidate_df = pd.DataFrame(candidate_points, columns=variable_names)

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(candidate_points)

    # Selección D-optimal
    selected_indices = []
    remaining = list(range(len(X_scaled)))
    for _ in range(n_experiments):
        best_idx = None
        best_score = -np.inf
        for idx in remaining:
            trial_set = selected_indices + [idx]
            X_trial = X_scaled[trial_set]
            score = calculate_d_criterion_stable(X_trial)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    # Crear DataFrame con resultados
    selected_df = candidate_df.iloc[selected_indices].copy()
    selected_df.insert(0, "No.", range(1, len(selected_df) + 1))
    selected_df["上面ダレ"] = ""
    selected_df["側面ダレ"] = ""
    selected_df["摩耗量"] = ""

    # Guardar Excel
    os.makedirs(output_folder, exist_ok=True)
    output_excel = os.path.join(output_folder, "selected_samples.xlsx")
    selected_df.to_excel(output_excel, index=False)

    # Guardar imagen
    fig_path = os.path.join(output_folder, "histogram_sample.png")
    plt.figure()
    for col in variable_names:
        plt.hist(candidate_df[col], alpha=0.5, label=col)
    plt.legend()
    plt.title("Histograma de variables")
    plt.savefig(fig_path)
    plt.close()

    viz_results = create_visualization(df_filtered, X, optimal_indices, output_prefix)

    return {
        'num_selected': len(optimal_indices),
        'best_criterion': best_criterion,
        'pca_variance_ratio': viz_results['pca_variance_ratio'],
        'feature_importance': viz_results['feature_importance'],
        'selected_dataframe': d_optimal_data
    }

    # return selected_df, fig_path
