import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from d_i_logic import (
    load_and_validate_existing_data,
    match_existing_experiments_enhanced,
    hierarchical_candidate_reduction,
    select_d_optimal_design_enhanced,
    select_i_optimal_design,
    visualize_pca_enhanced,
    visualize_umap_enhanced
)

def run_d_i_optimizer(sample_file, existing_data_file, output_folder, num_experiments=15):
    # ES: === Leer combinaciones ya generadas ===
    # EN: === Read already-generated combinations ===
    # JP: === 既に生成された組合せを読み込む ===
    candidate_df = pd.read_excel(sample_file)
    candidate_points = candidate_df.values
    variable_names = list(candidate_df.columns)

    # === Procesar datos existentes ===
    existing_data, _ = load_and_validate_existing_data(existing_data_file, candidate_df, verbose=False)
    existing_indices = []
    if existing_data is not None and len(existing_data) > 0:
        existing_indices = match_existing_experiments_enhanced(
            candidate_points, existing_data, variable_names
        )

    # === Reducción de candidatos si excede umbral ===
    if len(candidate_points) > 10000:
        candidate_points, mapping = hierarchical_candidate_reduction(
            candidate_points, 5000, existing_indices
        )
        if existing_indices:
            existing_indices = [mapping.index(i) for i in existing_indices if i in mapping]

        candidate_df = pd.DataFrame(candidate_points, columns=variable_names)

    # === Escalar datos ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(candidate_points)

    # === D最適化 & i最適化 ===
    d_indices, _ = select_d_optimal_design_enhanced(X_scaled, existing_indices, num_experiments, verbose=False)
    i_indices = select_i_optimal_design(X_scaled, num_experiments, existing_indices)

    # === Resultados como DataFrames ===
    selected_d_df = candidate_df.iloc[[i for i in d_indices if i not in existing_indices]].copy()
    selected_i_df = candidate_df.iloc[[i for i in i_indices if i not in existing_indices]].copy()

    selected_d_df.insert(0, "No.", range(1, len(selected_d_df) + 1))
    selected_i_df.insert(0, "No.", range(1, len(selected_i_df) + 1))

    selected_d_df["上面ダレ"] = ""
    selected_d_df["側面ダレ"] = ""
    selected_d_df["摩耗量"] = ""

    # ES: Guardar Excel | EN: Save Excel | JA: Excelを保存
    os.makedirs(output_folder, exist_ok=True)
    d_path = os.path.join(output_folder, "D_optimal_新規実験点.xlsx")
    i_path = os.path.join(output_folder, "I_optimal_新規実験点.xlsx")
    selected_d_df.to_excel(d_path, index=False)
    selected_i_df.to_excel(i_path, index=False)

    # === Visualización (PCA y UMAP) ===
    pca_d_path = os.path.join(output_folder, "D_optimal_新規実験点.png")
    pca_i_path = os.path.join(output_folder, "I_optimal_新規実験点.png")
    umap_path = os.path.join(output_folder, "UMAP_可視化.png")

    visualize_pca_enhanced(X_scaled, d_indices, existing_indices, output_path=pca_d_path)
    visualize_pca_enhanced(X_scaled, i_indices, existing_indices, output_path=pca_i_path)
    visualize_umap_enhanced(X_scaled, d_indices, i_indices, existing_indices, output_path=umap_path)

    return {
        "d_dataframe": selected_d_df,
        "i_dataframe": selected_i_df,
        "d_path": d_path,
        "i_path": i_path,
        "image_paths": [pca_d_path, pca_i_path, umap_path]
    }
