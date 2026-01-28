# d_i_logic.py

import pandas as pd
import numpy as np
from itertools import product
from sklearn.preprocessing import StandardScaler
from scipy.linalg import qr
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

# ES: === Carga de datos existentes ===
# EN: === Load existing data ===
# JA: === 既存データの読み込み ===
def load_and_validate_existing_data(existing_file, design_df, verbose=True):
    try:
        existing_df = pd.read_excel(existing_file)
        variable_names = design_df.columns.tolist() if "説明変数名" not in design_df else design_df["説明変数名"].tolist()

        missing_vars = []
        available_vars = []

        for var in variable_names:
            if var in existing_df.columns:
                available_vars.append(var)
            else:
                missing_vars.append(var)

        if len(available_vars) < len(variable_names) * 0.7:
            return None, []
        existing_explanatory = existing_df[available_vars]
        existing_explanatory = existing_explanatory.dropna().drop_duplicates()

        return existing_explanatory, available_vars
    except:
        return None, []

# ES: === Generación de candidatos ===
# EN: === Candidate generation ===
# JA: === 候補生成 ===
def generate_candidate_points(design_df):
    levels = []
    for _, row in design_df.iterrows():
        levels.append(np.arange(row["最小値"], row["最大値"] + row["刻み幅"], row["刻み幅"]))
    return np.array(list(product(*levels)))

# ES: === Emparejamiento con datos existentes ===
# EN: === Matching against existing data ===
# JA: === 既存データとのマッチング ===
def match_existing_experiments_enhanced(candidate_points, existing_data, variable_names,
                                        tolerance_relative=1e-6, tolerance_absolute=1e-8, verbose=False):
    if existing_data is None or len(existing_data) == 0:
        return []

    candidate_df = pd.DataFrame(candidate_points, columns=variable_names)
    matched_indices = []

    for exist_idx, exist_row in enumerate(existing_data.values):
        for cand_idx, cand_row in enumerate(candidate_df.values):
            match = True
            for var_idx in range(len(variable_names)):
                abs_error = abs(exist_row[var_idx] - cand_row[var_idx])
                if abs_error > tolerance_absolute:
                    rel_error = abs_error / abs(exist_row[var_idx]) if exist_row[var_idx] != 0 else 1
                    if rel_error > tolerance_relative:
                        match = False
                        break
            if match:
                matched_indices.append(cand_idx)
                break
    return list(set(matched_indices))

# ES: === Reducción de candidatos ===
# EN: === Candidate reduction ===
# JA: === 候補削減 ===
def hierarchical_candidate_reduction(candidate_points, max_candidates=5000, existing_indices=None):
    from sklearn.cluster import MiniBatchKMeans
    n_original = len(candidate_points)

    if n_original <= max_candidates:
        return candidate_points, list(range(n_original))

    if existing_indices:
        existing_set = set(existing_indices)
        available_indices = [i for i in range(n_original) if i not in existing_set]
        available_points = candidate_points[available_indices]
        n_to_select = max_candidates - len(existing_indices)
    else:
        available_indices = list(range(n_original))
        available_points = candidate_points
        n_to_select = max_candidates
        existing_indices = []

    if n_to_select <= 0:
        return candidate_points[existing_indices], existing_indices

    n_clusters = min(n_to_select, len(available_points))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(available_points)

    selected_indices = list(existing_indices)

    for i in range(n_clusters):
        cluster_mask = clusters == i
        if np.any(cluster_mask):
            cluster_points = available_points[cluster_mask]
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            closest_idx = np.argmin(distances)
            selected_indices.append(available_indices[np.where(cluster_mask)[0][closest_idx]])

    reduced_points = candidate_points[selected_indices]
    return reduced_points, selected_indices

# === D最適化 ===
def calculate_d_criterion_stable(X):
    try:
        q, r = qr(X, mode='economic')
        diag_r = np.diag(r)
        det = np.abs(np.prod(diag_r))
        log_det = np.log(det) if det > 1e-300 else -np.inf
        return log_det
    except:
        return -np.inf

def select_d_optimal_design_enhanced(X_all, existing_indices, new_experiments, verbose=False):
    base = list(existing_indices) if existing_indices else []
    remaining = [i for i in range(len(X_all)) if i not in base]
    total_select = len(base) + new_experiments
    selected = list(base)

    for _ in range(new_experiments):
        best_candidate = None
        best_score = -np.inf
        for idx in remaining:
            trial_set = selected + [idx]
            score = calculate_d_criterion_stable(X_all[trial_set])
            if score > best_score:
                best_score = score
                best_candidate = idx
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
    return selected, best_score

# ES: === i最適化 ===
# EN: === I-optimal selection ===
# JA: === I最適化 ===
def select_i_optimal_design(X_all, new_experiments, existing_indices=None):
    if existing_indices:
        selected_indices = list(existing_indices)
    else:
        selected_indices = [0]

    remaining_indices = [i for i in range(len(X_all)) if i not in selected_indices]
    target_total = len(selected_indices) + new_experiments

    while len(selected_indices) < target_total and remaining_indices:
        dists = cdist(X_all[remaining_indices], X_all[selected_indices])
        min_dists = dists.min(axis=1)
        next_idx = remaining_indices[np.argmax(min_dists)]
        selected_indices.append(next_idx)
        remaining_indices.remove(next_idx)

    # ES: Calcular el I-criterion (por ejemplo, la mínima distancia entre los seleccionados)
    # EN: Compute I-criterion (e.g., the minimum distance among selected points)
    # JA: I基準を計算（例：選択点間の最小距離）
    selected_X = X_all[selected_indices]
    dists_selected = cdist(selected_X, selected_X)
    np.fill_diagonal(dists_selected, np.inf)
    i_score = np.min(dists_selected)

    return selected_indices, i_score

# ES: === Visualización PCA ===
# EN: === PCA visualization ===
# JA: === PCA可視化 ===
def visualize_pca_enhanced(X_scaled, selected_indices, existing_indices, output_path="pca.png"):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], color="lightgray", label="All candidates")
    if existing_indices:
        plt.scatter(components[existing_indices, 0], components[existing_indices, 1], color="blue", label="Existing")
    plt.scatter(components[selected_indices, 0], components[selected_indices, 1], color="red", label="Selected")
    plt.title("PCA Visualization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ES: === Visualización UMAP ===
# EN: === UMAP visualization ===
# JA: === UMAP可視化 ===
def visualize_umap_enhanced(X_scaled, d_indices, i_indices, existing_indices, output_path="umap.png"):
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], color="lightgray", label="All candidates")
    if existing_indices:
        plt.scatter(embedding[existing_indices, 0], embedding[existing_indices, 1], color="blue", label="Existing")
    plt.scatter(embedding[d_indices, 0], embedding[d_indices, 1], color="red", label="D-optimal")
    plt.scatter(embedding[i_indices, 0], embedding[i_indices, 1], color="green", label="I-optimal")
    plt.title("UMAP Visualization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
