import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
from itertools import product, combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import qr
from scipy.spatial.distance import cdist
import time
import os
import json
import warnings
import shutil
from datetime import datetime
warnings.filterwarnings('ignore')

# ES: ✅ NUEVO: Configuración de joblib para evitar errores de subprocess en Windows | EN: ✅ NEW: joblib configuration to avoid subprocess errors on Windows | JA: ✅ 新規: Windowsでのsubprocessエラー回避のためjoblibを設定
try:
    import joblib
    # ES: Configurar joblib para usar un número fijo de workers y evitar detección automática de CPU
    # EN: Configure joblib to use a fixed number of workers and avoid automatic CPU detection
    # JA: joblibのワーカー数を固定し、CPU自動検出を回避
    joblib.parallel.BACKENDS['threading'].n_jobs = 1
    joblib.parallel.BACKENDS['multiprocessing'].n_jobs = 1
    print("✅ subprocess エラー回避のため joblib を設定しました")
except ImportError:
    print("⚠️ joblib が利用できないため、個別設定なしで続行します")

# ES: === Configuración de fuentes === | EN: === Font configuration === | JA: === フォント設定 ===
FONT_NAME = "Meiryo"
rcParams['font.family'] = FONT_NAME

# ES: Configuración de optimización
# EN: Optimization configuration
# JA: 最適化設定
USE_NUMERICAL_STABLE_METHOD = True
CANDIDATE_REDUCTION_THRESHOLD = 10000
MAX_REDUCED_CANDIDATES = 5000
VERBOSE = True

def load_and_validate_existing_data(existing_file, design_df, verbose=True):
    """ES: Carga y valida datos experimentales existentes
    EN: Load and validate existing experimental data
    JA: 既存の実験データを読み込み・検証
    """
    try:
        ext = os.path.splitext(str(existing_file))[1].lower()
        existing_df = pd.read_csv(existing_file, encoding="utf-8-sig") if ext == ".csv" else pd.read_excel(existing_file)
        if verbose:
            print(f"実験データ既存ファイル読み込み完了: {len(existing_df)} 行 × {len(existing_df.columns)} 列")
            print(f"ℹ️ 既存列: {list(existing_df.columns)}")

        # ES: Obtener nombres de variables explicativas
        # EN: Get explanatory variable names
        # JA: 説明変数名を取得
        # ES: `design_df` puede venir en formato "tabla de diseño" con columna '説明変数名'
        # EN: `design_df` may come as a "design table" with a '説明変数名' column
        # JA: `design_df` は「設計表」形式で '説明変数名' 列を含む場合がある
        if isinstance(design_df, pd.DataFrame) and "説明変数名" in design_df.columns:
            variable_names = design_df["説明変数名"].astype(str).tolist()
        else:
            variable_names = design_df.columns.tolist() if isinstance(design_df, pd.DataFrame) else list(design_df)
        if verbose:
            print(f"🎯 目的変数: {variable_names}")

        # ES: Extraer solo variables explicativas de datos existentes
        # EN: Extract only explanatory variables from existing data
        # JA: 既存データから説明変数のみ抽出
        missing_vars = []
        available_vars = []

        for var in variable_names:
            if var in existing_df.columns:
                available_vars.append(var)
            else:
                missing_vars.append(var)

        if missing_vars:
            print(f"⚠️ 以下の変数説明変数が実験データに見つかりません: {missing_vars}")
            if len(available_vars) < len(variable_names) * 0.7:
                print("❌ 実験データ不足 (70%未満) - 実験データを使用しない")
                return None, []
            else:
                print(f"✅ 利用可能な変数 ({len(available_vars)}/{len(variable_names)}) - 続行")

        # ES: Extraer solo variables explicativas
        # EN: Extract only explanatory variables
        # JA: 説明変数のみ抽出
        existing_explanatory = existing_df[available_vars]

        if verbose:
            print(f"✅ 変数説明変数抽出完了: {len(existing_explanatory)} 行 × {len(available_vars)} 列")
            print(f"📊 データサンプル (最初の3行):")
            print(existing_explanatory.head(3))
            print(f"📈 データ統計:")
            print(existing_explanatory.describe())

        # ES: Verificaciones de calidad específicas para procesos químicos
        # EN: Quality checks tailored for chemical processes
        # JA: 化学プロセス向けの品質チェック
        # ES: 1. Verificación de valores faltantes
        # EN: 1) Missing-value check
        # JA: 1) 欠損値チェック
        missing_count = existing_explanatory.isnull().sum().sum()
        if missing_count > 0:
            print(f"⚠️ 欠損値検出: {missing_count}")
            existing_explanatory = existing_explanatory.dropna()
            print(f"🔧 欠損値削除後: {len(existing_explanatory)} 行")

        # ES: 2. Verificación de puntos experimentales duplicados
        # EN: 2) Duplicate-point check
        # JA: 2) 重複点チェック
        duplicates = existing_explanatory.duplicated().sum()
        if duplicates > 0:
            print(f"⚠️ 実験データ重複検出: {duplicates}")
            existing_explanatory = existing_explanatory.drop_duplicates()
            print(f"🔧 重複削除後: {len(existing_explanatory)} 行")

        return existing_explanatory, available_vars

    except FileNotFoundError:
        print(f"❌ 実験データ既存ファイル見つかりません: {existing_file}")
        return None, []
    except Exception as e:
        print(f"❌ 実験データ既存読み込みエラー: {e}")
        return None, []

def match_existing_experiments_enhanced(candidate_points, existing_data, variable_names, 
                                      tolerance_relative=1e-6, tolerance_absolute=1e-8, verbose=True):
    """ES: Emparejamiento de alta precisión de condiciones experimentales químicas
    EN: High-precision matching of chemical experimental conditions
    JA: 化学実験条件の高精度マッチング
    """
    if existing_data is None or len(existing_data) == 0:
        return []

    print(f"🔍 実験データ既存点検索開始")
    print(f"  - 候補点数: {len(candidate_points):,}")
    print(f"  - 既存実験点数: {len(existing_data)}")
    print(f"  - 相対許容誤差: {tolerance_relative}")
    print(f"  - 絶対許容誤差: {tolerance_absolute}")

    # ES: Convertir puntos candidatos a DataFrame | EN: Convert candidate points to a DataFrame | JA: 候補点をDataFrameに変換
    candidate_df = pd.DataFrame(candidate_points, columns=variable_names)

    # ES: Estandarizar ambos conjuntos de datos | EN: Standardize both datasets | JA: 両データセットを標準化
    scaler = StandardScaler()
    candidate_scaled = scaler.fit_transform(candidate_df)

    # ES: Alinear datos existentes al mismo orden de variables | EN: Align existing data to the same variable order | JA: 既存データを変数順に合わせる
    existing_aligned = existing_data[variable_names]
    existing_scaled = scaler.transform(existing_aligned)

    matched_indices = []
    match_details = []

    # ES: Para cada punto experimental existente, buscar el candidato más cercano
    # EN: For each existing experimental point, find the nearest candidate
    # JA: 各既存実験点に対して最も近い候補点を探索
    for exist_idx, exist_row in enumerate(existing_aligned.values):
        min_distance = float('inf')
        best_candidate_idx = None

        for cand_idx, cand_row in enumerate(candidate_df.values):
            # ES: Comparación basada en error relativo | EN: Comparison based on relative error | JA: 相対誤差に基づく比較
            relative_errors = []
            absolute_ok = True

            for var_idx, var_name in enumerate(variable_names):
                exist_val = exist_row[var_idx]
                cand_val = cand_row[var_idx]

                # ES: Verificación de error absoluto | EN: Absolute-error check | JA: 絶対誤差チェック
                abs_error = abs(exist_val - cand_val)
                if abs_error > tolerance_absolute:
                    # ES: También verificar error relativo | EN: Also check relative error | JA: 相対誤差も確認
                    if exist_val != 0:
                        rel_error = abs_error / abs(exist_val)
                        if rel_error > tolerance_relative:
                            absolute_ok = False
                            break
                    else:
                        absolute_ok = False
                        break

                relative_errors.append(abs_error)

            if absolute_ok:
                # Distancia total (en espacio estandarizado)
                distance = np.linalg.norm(existing_scaled[exist_idx] - candidate_scaled[cand_idx])

                if distance < min_distance:
                    min_distance = distance
                    best_candidate_idx = cand_idx

        if best_candidate_idx is not None:
            matched_indices.append(best_candidate_idx)

            # ES: Registrar detalles del emparejamiento | EN: Record matching details | JA: マッチング詳細を記録
            match_detail = {
                'Número_experimento_existente': exist_idx,
                'Número_punto_candidato': best_candidate_idx,
                'Distancia': min_distance,
                'Condiciones_experimento_existente': existing_aligned.iloc[exist_idx].to_dict(),
                'Condiciones_punto_candidato': candidate_df.iloc[best_candidate_idx].to_dict()
            }
            match_details.append(match_detail)

            if verbose and len(matched_indices) <= 5:
                print(f"✅ マッチング {len(matched_indices)}: Existente#{exist_idx} → Candidato#{best_candidate_idx} (distancia: {min_distance:.4f})")

    # ES: Eliminar duplicados
    # EN: Remove duplicates
    # JA: 重複を除去
    unique_matched = list(set(matched_indices))

    print(f"📊 マッチング結果:")
    print(f"  - 初期マッチング: {len(matched_indices)}")
    print(f"  - 重複削除後: {len(unique_matched)}")
    print(f"  - マッチング率: {len(unique_matched)/len(existing_data)*100:.1f}%")

    if len(unique_matched) == 0:
        print("⚠️ 既存実験点マッチング見つかりません")
        print("💡 考えられる原因:")
        print("  1. 既存実験点条件既存範囲外")
        print("  2. ステップ設定既存データに一致しない")
        print("  3. 許容誤差設定厳しすぎ")

        # ES: Proporcionar información de diagnóstico
        # EN: Provide diagnostic information
        # JA: 診断情報を出力
        print("\n🔍 診断情報:")
        for var in variable_names:
            exist_range = (existing_aligned[var].min(), existing_aligned[var].max())
            cand_range = (candidate_df[var].min(), candidate_df[var].max())
            print(f"  {var}: Existente{exist_range} vs Candidato{cand_range}")

    return unique_matched

def hierarchical_candidate_reduction(candidate_points, max_candidates=5000, existing_indices=None):
    """ES: Reducción de candidatos mediante muestreo jerárquico
    EN: Reduce candidates via hierarchical sampling
    JA: 階層的サンプリングによる候補の削減"""
    n_original = len(candidate_points)

    if n_original <= max_candidates:
        print(f"📊 候補点数 ({n_original:,}) 要約不要 (閾値: {max_candidates:,})")
        return candidate_points, list(range(n_original))

    print(f"🔄 ✅ 階層的サンプリング実行: {n_original:,} → {max_candidates:,} 点")

    # ES: Proteger puntos experimentales existentes | EN: Preserve existing experimental points | JA: 既存実験点を保護
    if existing_indices:
        existing_set = set(existing_indices)
        available_indices = [i for i in range(n_original) if i not in existing_set]
        available_points = candidate_points[available_indices]
        n_to_select = max_candidates - len(existing_indices)
        print(f"📍 既存実験点保持: {len(existing_indices)} 点")
    else:
        available_indices = list(range(n_original))
        available_points = candidate_points
        n_to_select = max_candidates
        existing_indices = []

    if n_to_select <= 0:
        print("⚠️ 既存点のみで上限に達")
        return candidate_points[existing_indices], existing_indices

    print(f"🎯 新選択目標: {n_to_select:,} 点")

    try:
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = min(n_to_select, len(available_points))
        print(f"🔧 MiniBatchKMeans Clustering: {n_clusters} clusters")

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            batch_size=min(1000, len(available_points)//10),
            n_init=3,
            max_iter=100
        )

        start_time = time.time()
        # ES: scikit-learn reciente ya no soporta n_jobs en MiniBatchKMeans. Para replicar n_jobs=1 limitamos threads SOLO durante el fit.
        # EN: Recent scikit-learn no longer supports n_jobs in MiniBatchKMeans. To mimic n_jobs=1 we limit threads ONLY during fit.
        # JA: 最近のscikit-learnはMiniBatchKMeansでn_jobsをサポートしない。n_jobs=1相当のためfit中のみスレッドを制限。
        # ES: Además, en algunos entornos Windows recientes `joblib/loky` intenta usar `wmic` para contar cores y puede fallar.
        # EN: Also, on some recent Windows environments `joblib/loky` tries to use `wmic` to count cores and can fail.
        # JA: さらに、最近のWindows環境では `joblib/loky` がコア数取得に `wmic` を使おうとして失敗することがあります。
        # ES: Para evitarlo, forzamos backend "threading" SOLO en este fit.
        # EN: To avoid that, we force the "threading" backend ONLY for this fit.
        # JA: 回避のため、このfitの間だけ "threading" バックエンドを強制します。
        try:
            from threadpoolctl import threadpool_limits
            # ES: Silenciar/evitar detección de cores físicos vía wmic en loky (Windows). Mantiene el algoritmo; limita a 1 core en este bloque.
            # EN: Suppress physical-core detection via wmic in loky (Windows). Keeps algorithm; limits to 1 core in this block.
            # JA: loky（Windows）でのwmicによる物理コア検出を抑制。アルゴリズムは維持し、このブロック内は1コアに制限。
            _prev_loky_max_cpu = os.environ.get("LOKY_MAX_CPU_COUNT")
            os.environ["LOKY_MAX_CPU_COUNT"] = "1"
            try:
                import joblib
                with joblib.parallel_backend("threading", n_jobs=1):
                    with threadpool_limits(limits=1):
                        clusters = kmeans.fit_predict(available_points)
            except Exception:
                with threadpool_limits(limits=1):
                    clusters = kmeans.fit_predict(available_points)
            finally:
                if _prev_loky_max_cpu is None:
                    os.environ.pop("LOKY_MAX_CPU_COUNT", None)
                else:
                    os.environ["LOKY_MAX_CPU_COUNT"] = _prev_loky_max_cpu
        except Exception:
            # ES: Si threadpoolctl no está disponible u ocurre cualquier problema, continuar sin limitar threads
            # EN: If threadpoolctl is unavailable or anything fails, proceed without limiting threads
            # JA: threadpoolctlが無い/失敗した場合はスレッド制限なしで続行
            clusters = kmeans.fit_predict(available_points)
        clustering_time = time.time() - start_time
        print(f"⏱️ クラスタリング時間: {clustering_time:.2f} 秒")

        # ES: Seleccionar punto representativo de cada cluster | EN: Pick one representative point per cluster | JA: クラスタごとに代表点を選択
        selected_indices = list(existing_indices)

        for i in range(n_clusters):
            cluster_mask = clusters == i
            if np.any(cluster_mask):
                cluster_indices_in_available = np.where(cluster_mask)[0]
                cluster_original_indices = [available_indices[j] for j in cluster_indices_in_available]

                # ES: Seleccionar punto más cercano al centro del cluster | EN: Pick the point closest to the cluster center | JA: クラスタ中心に最も近い点を選択
                cluster_points = available_points[cluster_mask]
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_points - center, axis=1)
                closest_idx_in_cluster = np.argmin(distances)
                closest_original_idx = cluster_original_indices[closest_idx_in_cluster]

                selected_indices.append(closest_original_idx)

        reduced_points = candidate_points[selected_indices]

        print(f"✅ 階層的サンプリング完了: 最終候補点数 {len(reduced_points):,}")
        print(f"  - 既存実験点保持: {len(existing_indices)} 点")
        print(f"  - 新選択点数: {len(selected_indices) - len(existing_indices)} 点")

        return reduced_points, selected_indices

    except Exception as e:
        print(f"⚠️ 階層的サンプリングエラー: {e}")
        return candidate_points, list(range(len(candidate_points)))

def calculate_d_criterion_stable(X, method='auto'):
    """ES: Cálculo numéricamente estable del criterio D
    EN: Numerically stable computation of the D criterion
    JA: D基準の数値的に安定した計算"""
    try:
        condition_number = np.linalg.cond(X)

        if USE_NUMERICAL_STABLE_METHOD or method == 'auto' and condition_number > 1e12:
            method = 'svd'
            if VERBOSE and condition_number > 1e12:
                print(f"🔧 高条件検出 ({condition_number:.2e}) - SVDメソッド適用")

        if method == 'svd':
            _, s, _ = np.linalg.svd(X, full_matrices=False)
            valid_singular_values = s[s > 1e-14]
            if len(valid_singular_values) == 0:
                return -np.inf, condition_number
            log_det = np.sum(np.log(valid_singular_values))
        else:
            q, r = qr(X, mode='economic')
            diag_r = np.diag(r)
            det = np.abs(np.prod(diag_r))
            log_det = np.log(det) if det > 1e-300 else -np.inf

        return log_det, condition_number

    except Exception as e:
        if VERBOSE:
            print(f"⚠️ 基準D計算エラー: {e}")
        return -np.inf, np.inf

def select_d_optimal_design_enhanced(X_all, existing_indices, new_experiments, verbose=True):
    """ES: Selección de diseño D-óptimo (puntos experimentales existentes + nuevos)
    EN: D-optimal design selection (existing experimental points + new ones)
    JA: D最適設計の選択（既存実験点＋新規）"""
    base = list(existing_indices) if existing_indices else []
    remaining = [i for i in range(len(X_all)) if i not in base]
    total_select = len(base) + new_experiments

    if verbose:
        print(f"  - 既存実験点数: {len(base)} 点")
        print(f"  - 新規実験点数: {new_experiments} 点")
        print(f"  - 選択点数合計: {total_select} 点")

    if new_experiments <= 0:
        if verbose:
            print(f"  ✅ 既存実験点のみ完了")
        score, _ = calculate_d_criterion_stable(X_all[base])
        return base, score

    selected = list(base)

    for step in range(new_experiments):
        best_candidate = None
        best_score = -np.inf

        # Para datos grandes, usar muestreo
        if len(remaining) > 1000:
            sample_size = min(500, len(remaining))
            sample_indices = np.random.choice(remaining, sample_size, replace=False)
            candidates_to_check = sample_indices
        else:
            candidates_to_check = remaining

        for idx in candidates_to_check:
            trial_set = selected + [idx]
            X_subset = X_all[trial_set]
            score, condition_num = calculate_d_criterion_stable(X_subset)

            if score > best_score:
                best_score = score
                best_candidate = idx

        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            if verbose:
                print(f"  ✅ 新選択 {step+1}/{new_experiments}: 点{best_candidate}, スコア: {best_score:.4f}")
        else:
            if verbose:
                print(f"  ⚠️ ステップ {step+1} 適切な候補見つかりません")
            break

    final_score, final_condition = calculate_d_criterion_stable(X_all[selected])
    return selected, final_score

def select_i_optimal_design(X_all, new_experiments, existing_indices=None):
    """ES: Selección de diseño I-óptimo (puntos experimentales existentes + nuevos)
    EN: I-optimal design selection (existing experimental points + new ones)
    JA: I最適設計の選択（既存実験点＋新規）"""
    if existing_indices:
        selected_indices = list(existing_indices)
        print(f"  - 既存実験点数: {len(existing_indices)} 点")
        print(f"  - 新規実験点数: {new_experiments} 点")
        print(f"  - 選択点数合計: {len(existing_indices) + new_experiments} 点")
    else:
        selected_indices = []
        print(f"  - 新規実験点数: {new_experiments} 点 (既存点なし)")

    remaining_indices = [i for i in range(len(X_all)) if i not in selected_indices]
    target_total = len(selected_indices) + new_experiments

    step = 0
    while len(selected_indices) < target_total and remaining_indices:
        if len(selected_indices) == 0:
            # ES: Si no hay puntos seleccionados, elegir el primer punto disponible
            # EN: If no points selected yet, pick the first available point
            # JA: 選択点が無い場合は最初の利用可能点を選ぶ
            next_index = remaining_indices[0]
            selected_indices.append(next_index)
            remaining_indices.remove(next_index)
            step += 1
            print(f"  ✅ 新選択 {step}/{new_experiments}: 点{next_index} (最初の点)")
        else:
            # ES: Calcular distancias solo si hay puntos seleccionados
            # EN: Compute distances only when there are selected points
            # JA: 選択点がある場合のみ距離を計算
            dists = cdist(X_all[remaining_indices], X_all[selected_indices])
            min_dists = dists.min(axis=1)
            next_idx_in_remaining = np.argmax(min_dists)
            next_index = remaining_indices[next_idx_in_remaining]
            selected_indices.append(next_index)
            remaining_indices.remove(next_index)
            step += 1
            print(f"  ✅ 新選択 {step}/{new_experiments}: 点{next_index}")

    return selected_indices

def visualize_feature_histograms(candidate_df, d_indices, i_indices, existing_indices, variable_names, output_folder, optimization_type="both"):
    """ES: 📊 Histogramas de características con colores diferenciados (uno por variable)
    EN: 📊 Feature histograms with distinct colors (one per variable)
    JA: 📊 特徴量ヒストグラム（変数ごとに色分け）"""
    print(f"\n📊 特徴量分布の可視化開始... (最適化タイプ: {optimization_type})")

    image_paths = []
    for var_name in variable_names:
        plt.figure(figsize=(6, 4))

        # Histograma de todos los puntos candidatos (fondo)
        plt.hist(candidate_df[var_name], bins=30, alpha=0.3, color='lightgray', 
                label=f'全候補点 ({len(candidate_df)})', density=True)

        # ES: Puntos experimentales existentes | EN: Existing experimental points | JA: 既存実験点
        if existing_indices:
            existing_values = candidate_df.iloc[existing_indices][var_name]
            plt.hist(existing_values, bins=15, alpha=0.8, color='blue', 
                    label=f'既存点 ({len(existing_indices)})', density=True)

        # ES: Mostrar solo los datos relevantes según el tipo de optimización | EN: Show only data relevant to the optimization type | JA: 最適化タイプに関連するデータのみ表示
        if optimization_type in ["d", "D", "d_optimal"]:
            # ES: Solo mostrar datos D-óptimo | EN: Show only D-optimal data | JA: D最適データのみ表示
            d_new_indices = [idx for idx in d_indices if idx not in existing_indices]
            if d_new_indices:
                d_values = candidate_df.iloc[d_new_indices][var_name]
                plt.hist(d_values, bins=10, alpha=0.8, color='red', 
                        label=f'D-最適新規点 ({len(d_new_indices)})', density=True)
        elif optimization_type in ["i", "I", "i_optimal"]:
            # ES: Solo mostrar datos I-óptimo | EN: Show only I-optimal data | JA: I最適データのみ表示
            i_new_indices = [idx for idx in i_indices if idx not in existing_indices]
            if i_new_indices:
                i_values = candidate_df.iloc[i_new_indices][var_name]
                plt.hist(i_values, bins=10, alpha=0.8, color='green', 
                        label=f'I-最適新規点 ({len(i_new_indices)})', density=True)
        else:
            # ES: Mostrar ambos (comportamiento original) | EN: Show both (original behavior) | JA: 両方表示（元の挙動）
            d_new_indices = [idx for idx in d_indices if idx not in existing_indices]
            if d_new_indices:
                d_values = candidate_df.iloc[d_new_indices][var_name]
                plt.hist(d_values, bins=10, alpha=0.8, color='red', 
                        label=f'D-最適新規点 ({len(d_new_indices)})', density=True)

            i_new_indices = [idx for idx in i_indices if idx not in existing_indices]
            if i_new_indices:
                i_values = candidate_df.iloc[i_new_indices][var_name]
                plt.hist(i_values, bins=10, alpha=0.8, color='green', 
                        label=f'I-最適新規点 ({len(i_new_indices)})', density=True)

        # ES: Ajustar título según el tipo de optimización | EN: Adjust title based on optimization type | JA: 最適化タイプに応じてタイトル調整
        if optimization_type in ["d", "D", "d_optimal"]:
            plt.title(f'{var_name}の分布 (D最適化)', fontsize=12, weight='bold')
        elif optimization_type in ["i", "I", "i_optimal"]:
            plt.title(f'{var_name}の分布 (I最適化)', fontsize=12, weight='bold')
        else:
            plt.title(f'{var_name}の分布', fontsize=12, weight='bold')
            
        plt.xlabel(var_name)
        plt.ylabel('密度')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # ES: Guardar imagen individual con sufijo según el tipo de optimización
        # EN: Save per-feature histogram with suffix based on optimization type
        # JA: 最適化タイプに応じた接尾辞でヒストグラムを保存
        safe_var_name = str(var_name).replace('/', '_').replace(' ', '_')
        if optimization_type in ["d", "D", "d_optimal"]:
            hist_path = os.path.join(output_folder, f"hist_D_{safe_var_name}.png")
        elif optimization_type in ["i", "I", "i_optimal"]:
            hist_path = os.path.join(output_folder, f"hist_I_{safe_var_name}.png")
        else:
            hist_path = os.path.join(output_folder, f"hist_{safe_var_name}.png")
            
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths.append(hist_path)
        print(f"✅ ヒストグラム保存完了: {hist_path}")
    print(f"✅ 特徴量分布の可視化完了 ({len(image_paths)} ヒストグラム)")
    return image_paths

def visualize_separate_dimension_reduction(X_scaled, d_indices, i_indices, existing_indices, variable_names, output_folder, optimization_type="both", selected_d_df=None, selected_i_df=None):
    """ES: 📈 Visualización de reducción de dimensionalidad separada (PCA y UMAP individuales) con números de muestra
    EN: 📈 Separate dimensionality-reduction visualization (individual PCA and UMAP) with sample numbers
    JA: 📈 次元削減可視化（PCA/UMAPを個別）サンプル番号付き
    """
    print(f"\n📈 次元削減可視化開始... (最適化タイプ: {optimization_type})")
    
    image_paths = []
    
    try:
        import umap
        
        # ES: Parámetros UMAP optimizados | EN: Tuned UMAP parameters | JA: 最適化済みUMAPパラメータ
        best_params = {"n_neighbors": 15, "min_dist": 0.1}
        
        # ES: Ejecutar UMAP | EN: Run UMAP | JA: UMAPを実行
        print(f"🔧 UMAP実行中...")
        reducer = umap.UMAP(
            n_neighbors=best_params["n_neighbors"], 
            min_dist=best_params["min_dist"], 
            n_components=2, 
            random_state=42,
            verbose=False
        )
        
        start_time = time.time()
        reduced_umap = reducer.fit_transform(X_scaled)
        umap_time = time.time() - start_time
        print(f"⏱️ UMAP実行時間: {umap_time:.2f} 秒")
        
        # ES: Ejecutar PCA | EN: Run PCA | JA: PCAを実行
        pca = PCA(n_components=2, random_state=42)
        reduced_pca = pca.fit_transform(X_scaled)
        
        # ES: === GRÁFICO PCA SEPARADO === | EN: === Separate PCA plot === | JA: === PCAグラフ（個別） ===
        plt.figure(figsize=(12, 8))
        
        # ES: Todos los candidatos (fondo) | EN: All candidates (background) | JA: 全候補（背景）
        plt.scatter(reduced_pca[:, 0], reduced_pca[:, 1], alpha=0.2, s=8, color='lightgray', label='候補点')
        
        # ES: Puntos experimentales existentes | EN: Existing experimental points | JA: 既存実験点
        if existing_indices:
            existing_pca = reduced_pca[existing_indices]
            plt.scatter(existing_pca[:, 0], existing_pca[:, 1], 
                       s=120, color='blue', alpha=0.9, marker='o', 
                       edgecolors='navy', linewidth=2, zorder=10,
                       label=f'既存点 ({len(existing_indices)})')
        
        # ES: Mostrar solo los datos relevantes según el tipo de optimización | EN: Show only data relevant to the optimization type | JA: 最適化タイプに関連するデータのみ表示
        if optimization_type in ["d", "D", "d_optimal"]:
            # ES: Solo mostrar datos D-óptimo | EN: Show only D-optimal data | JA: D最適データのみ表示
            d_new = [idx for idx in d_indices if idx not in existing_indices]
            if d_new:
                d_pca = reduced_pca[d_new]
                plt.scatter(d_pca[:, 0], d_pca[:, 1], 
                           s=100, marker='x', color='red', linewidth=3, 
                           zorder=8, label=f'D-最適新規点 ({len(d_new)})')
                
                # ES: Añadir números de muestra en puntos D-óptimo | EN: Add sample numbers on D-optimal points | JA: D最適点にサンプル番号を付与
                if selected_d_df is not None and 'No.' in selected_d_df.columns:
                    for i, (x, y) in enumerate(d_pca):
                        sample_num = selected_d_df.iloc[i]['No.']
                        plt.annotate(f'{sample_num}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, 
                                   color='red', weight='bold', zorder=12)
        elif optimization_type in ["i", "I", "i_optimal"]:
            # ES: Solo mostrar datos I-óptimo | EN: Show only I-optimal data | JA: I最適データのみ表示
            i_new = [idx for idx in i_indices if idx not in existing_indices]
            if i_new:
                i_pca = reduced_pca[i_new]
                plt.scatter(i_pca[:, 0], i_pca[:, 1], 
                           s=100, marker='^', color='green', 
                           zorder=8, label=f'I-最適新規点 ({len(i_new)})')
                
                # ES: Añadir números de muestra en puntos I-óptimo | EN: Add sample numbers on I-optimal points | JA: I最適点にサンプル番号を付与
                if selected_i_df is not None and 'No.' in selected_i_df.columns:
                    for i, (x, y) in enumerate(i_pca):
                        sample_num = selected_i_df.iloc[i]['No.']
                        plt.annotate(f'{sample_num}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, 
                                   color='green', weight='bold', zorder=12)
        else:
            # ES: Mostrar ambos (comportamiento original) | EN: Show both (original behavior) | JA: 両方表示（元の挙動）
            d_new = [idx for idx in d_indices if idx not in existing_indices]
            if d_new:
                d_pca = reduced_pca[d_new]
                plt.scatter(d_pca[:, 0], d_pca[:, 1], 
                           s=100, marker='x', color='red', linewidth=3, 
                           zorder=8, label=f'D-最適新規点 ({len(d_new)})')
                
                # ES: Añadir números de muestra en puntos D-óptimo | EN: Add sample numbers on D-optimal points | JA: D最適点にサンプル番号を付与
                if selected_d_df is not None and 'No.' in selected_d_df.columns:
                    for i, (x, y) in enumerate(d_pca):
                        sample_num = selected_d_df.iloc[i]['No.']
                        plt.annotate(f'{sample_num}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, 
                                   color='red', weight='bold', zorder=12)
            
            i_new = [idx for idx in i_indices if idx not in existing_indices]
            if i_new:
                i_pca = reduced_pca[i_new]
                plt.scatter(i_pca[:, 0], i_pca[:, 1], 
                           s=100, marker='^', color='green', 
                           zorder=8, label=f'I-最適新規点 ({len(i_new)})')
                
                # ES: Añadir números de muestra en puntos I-óptimo | EN: Add sample numbers on I-optimal points | JA: I最適点にサンプル番号を付与
                if selected_i_df is not None and 'No.' in selected_i_df.columns:
                    for i, (x, y) in enumerate(i_pca):
                        sample_num = selected_i_df.iloc[i]['No.']
                        plt.annotate(f'{sample_num}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, 
                                   color='green', weight='bold', zorder=12)
        
        # ES: Ajustar título según el tipo de optimización | EN: Adjust title based on optimization type | JA: 最適化タイプに応じてタイトル調整
        if optimization_type in ["d", "D", "d_optimal"]:
            plt.title('主成分分析 (PCA) 次元削減 - D最適化', fontsize=16, weight='bold')
        elif optimization_type in ["i", "I", "i_optimal"]:
            plt.title('主成分分析 (PCA) 次元削減 - I最適化', fontsize=16, weight='bold')
        else:
            plt.title('主成分分析 (PCA) 次元削減', fontsize=16, weight='bold')
            
        plt.xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # ES: Guardar PCA con sufijo según el tipo de optimización | EN: Save PCA with suffix based on optimization type | JA: 最適化タイプに応じた接尾辞でPCAを保存
        if optimization_type in ["d", "D", "d_optimal"]:
            pca_path = os.path.join(output_folder, "reduccion_dimensionalidad_pca_D.png")
        elif optimization_type in ["i", "I", "i_optimal"]:
            pca_path = os.path.join(output_folder, "reduccion_dimensionalidad_pca_I.png")
        else:
            pca_path = os.path.join(output_folder, "reduccion_dimensionalidad_pca.png")
            
        plt.savefig(pca_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths.append(pca_path)
        print(f"✅ PCA保存完了: {pca_path}")
        
        # ES: === GRÁFICO UMAP SEPARADO === | EN: === Separate UMAP plot === | JA: === UMAPグラフ（個別） ===
        plt.figure(figsize=(12, 8))
        
        # ES: Todos los candidatos (fondo) | EN: All candidates (background) | JA: 全候補（背景）
        plt.scatter(reduced_umap[:, 0], reduced_umap[:, 1], alpha=0.2, s=8, color='lightgray', label='候補点')
        
        # ES: Puntos experimentales existentes | EN: Existing experimental points | JA: 既存実験点
        if existing_indices:
            existing_umap = reduced_umap[existing_indices]
            plt.scatter(existing_umap[:, 0], existing_umap[:, 1], 
                       s=120, color='blue', alpha=0.9, marker='o', 
                       edgecolors='navy', linewidth=2, zorder=10,
                       label=f'既存点 ({len(existing_indices)})')
            
            # ES: Mostrar números en puntos existentes (primeros 10) | EN: Show numbers on existing points (first 10) | JA: 既存点に番号表示（先頭10点）
            for i, (x, y) in enumerate(existing_umap[:min(10, len(existing_umap))]):
                plt.annotate(f'{i+1}', (x, y), xytext=(3, 3), 
                           textcoords='offset points', fontsize=8, 
                           color='darkblue', weight='bold', zorder=11)
        
        # ES: Mostrar solo los datos relevantes según el tipo de optimización | EN: Show only data relevant to the optimization type | JA: 最適化タイプに関連するデータのみ表示
        if optimization_type in ["d", "D", "d_optimal"]:
            # ES: Solo mostrar datos D-óptimo | EN: Show only D-optimal data | JA: D最適データのみ表示
            d_new = [idx for idx in d_indices if idx not in existing_indices]
            if d_new:
                d_umap = reduced_umap[d_new]
                plt.scatter(d_umap[:, 0], d_umap[:, 1], 
                           s=100, marker='x', color='red', linewidth=3, 
                           zorder=8, label=f'D-最適新規点 ({len(d_new)})')
                
                # ES: Añadir números de muestra en puntos D-óptimo | EN: Add sample numbers on D-optimal points | JA: D最適点にサンプル番号を付与
                if selected_d_df is not None and 'No.' in selected_d_df.columns:
                    for i, (x, y) in enumerate(d_umap):
                        sample_num = selected_d_df.iloc[i]['No.']
                        plt.annotate(f'{sample_num}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, 
                                   color='red', weight='bold', zorder=12)
        elif optimization_type in ["i", "I", "i_optimal"]:
            # ES: Solo mostrar datos I-óptimo | EN: Show only I-optimal data | JA: I最適データのみ表示
            i_new = [idx for idx in i_indices if idx not in existing_indices]
            if i_new:
                i_umap = reduced_umap[i_new]
                plt.scatter(i_umap[:, 0], i_umap[:, 1], 
                           s=100, marker='^', color='green', 
                           zorder=8, label=f'I-最適新規点 ({len(i_new)})')
                
                # ES: Añadir números de muestra en puntos I-óptimo | EN: Add sample numbers on I-optimal points | JA: I最適点にサンプル番号を付与
                if selected_i_df is not None and 'No.' in selected_i_df.columns:
                    for i, (x, y) in enumerate(i_umap):
                        sample_num = selected_i_df.iloc[i]['No.']
                        plt.annotate(f'{sample_num}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, 
                                   color='green', weight='bold', zorder=12)
        else:
            # ES: Mostrar ambos (comportamiento original) | EN: Show both (original behavior) | JA: 両方表示（元の挙動）
            d_new = [idx for idx in d_indices if idx not in existing_indices]
            if d_new:
                d_umap = reduced_umap[d_new]
                plt.scatter(d_umap[:, 0], d_umap[:, 1], 
                           s=100, marker='x', color='red', linewidth=3, 
                           zorder=8, label=f'D-最適新規点 ({len(d_new)})')
                
                # ES: Añadir números de muestra en puntos D-óptimo | EN: Add sample numbers on D-optimal points | JA: D最適点にサンプル番号を付与
                if selected_d_df is not None and 'No.' in selected_d_df.columns:
                    for i, (x, y) in enumerate(d_umap):
                        sample_num = selected_d_df.iloc[i]['No.']
                        plt.annotate(f'{sample_num}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, 
                                   color='red', weight='bold', zorder=12)
            
            i_new = [idx for idx in i_indices if idx not in existing_indices]
            if i_new:
                i_umap = reduced_umap[i_new]
                plt.scatter(i_umap[:, 0], i_umap[:, 1], 
                           s=100, marker='^', color='green', 
                           zorder=8, label=f'I-最適新規点 ({len(i_new)})')
                
                # ES: Añadir números de muestra en puntos I-óptimo | EN: Add sample numbers on I-optimal points | JA: I最適点にサンプル番号を付与
                if selected_i_df is not None and 'No.' in selected_i_df.columns:
                    for i, (x, y) in enumerate(i_umap):
                        sample_num = selected_i_df.iloc[i]['No.']
                        plt.annotate(f'{sample_num}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, 
                                   color='green', weight='bold', zorder=12)
        
        # ES: Ajustar título según el tipo de optimización | EN: Adjust title based on optimization type | JA: 最適化タイプに応じてタイトル調整
        if optimization_type in ["d", "D", "d_optimal"]:
            plt.title('UMAP 次元削減 - D最適化', fontsize=16, weight='bold')
        elif optimization_type in ["i", "I", "i_optimal"]:
            plt.title('UMAP 次元削減 - I最適化', fontsize=16, weight='bold')
        else:
            plt.title('UMAP 次元削減', fontsize=16, weight='bold')
            
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # ES: Guardar UMAP con sufijo según el tipo de optimización | EN: Save UMAP with suffix based on optimization type | JA: 最適化タイプに応じた接尾辞でUMAPを保存
        if optimization_type in ["d", "D", "d_optimal"]:
            umap_path = os.path.join(output_folder, "reduccion_dimensionalidad_umap_D.png")
        elif optimization_type in ["i", "I", "i_optimal"]:
            umap_path = os.path.join(output_folder, "reduccion_dimensionalidad_umap_I.png")
        else:
            umap_path = os.path.join(output_folder, "reduccion_dimensionalidad_umap.png")
            
        plt.savefig(umap_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths.append(umap_path)
        print(f"✅ UMAP保存完了: {umap_path}")
        
        print(f"✅ 次元削減可視化完了 ({len(image_paths)} グラフ)")
        return image_paths
        
    except ImportError:
        print("❌ UMAP未インストール - PCAのみ表示")
        # ES: Solo PCA como respaldo | EN: Fallback to PCA only | JA: 代替としてPCAのみ実行
        pca = PCA(n_components=2, random_state=42)
        reduced_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(reduced_pca[:, 0], reduced_pca[:, 1], alpha=0.2, s=8, color='lightgray', label='候補点')
        
        if existing_indices:
            existing_pca = reduced_pca[existing_indices]
            plt.scatter(existing_pca[:, 0], existing_pca[:, 1], 
                       s=120, color='blue', alpha=0.9, marker='o', 
                       edgecolors='navy', linewidth=2, zorder=10,
                       label=f'既存点 ({len(existing_indices)})')
        
        # ES: Mostrar solo los datos relevantes según el tipo de optimización | EN: Show only data relevant to the optimization type | JA: 最適化タイプに関連するデータのみ表示
        if optimization_type in ["d", "D", "d_optimal"]:
            d_new = [idx for idx in d_indices if idx not in existing_indices]
            if d_new:
                d_pca = reduced_pca[d_new]
                plt.scatter(d_pca[:, 0], d_pca[:, 1], 
                           s=100, marker='x', color='red', linewidth=3, 
                           zorder=8, label=f'D-最適新規点 ({len(d_new)})')
                
                # ES: Añadir números de muestra en puntos D-óptimo | EN: Add sample numbers on D-optimal points | JA: D最適点にサンプル番号を付与
                if selected_d_df is not None and 'No.' in selected_d_df.columns:
                    for i, (x, y) in enumerate(d_pca):
                        sample_num = selected_d_df.iloc[i]['No.']
                        plt.annotate(f'{sample_num}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, 
                                   color='red', weight='bold', zorder=12)
        elif optimization_type in ["i", "I", "i_optimal"]:
            i_new = [idx for idx in i_indices if idx not in existing_indices]
            if i_new:
                i_pca = reduced_pca[i_new]
                plt.scatter(i_pca[:, 0], i_pca[:, 1], 
                           s=100, marker='^', color='green', 
                           zorder=8, label=f'I-最適新規点 ({len(i_new)})')
                
                # ES: Añadir números de muestra en puntos I-óptimo | EN: Add sample numbers on I-optimal points | JA: I最適点にサンプル番号を付与
                if selected_i_df is not None and 'No.' in selected_i_df.columns:
                    for i, (x, y) in enumerate(i_pca):
                        sample_num = selected_i_df.iloc[i]['No.']
                        plt.annotate(f'{sample_num}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, 
                                   color='green', weight='bold', zorder=12)
        
        # ES: Ajustar título según el tipo de optimización | EN: Adjust title based on optimization type | JA: 最適化タイプに応じてタイトル調整
        if optimization_type in ["d", "D", "d_optimal"]:
            plt.title('主成分分析 (PCA) 次元削減 - D最適化', fontsize=16, weight='bold')
        elif optimization_type in ["i", "I", "i_optimal"]:
            plt.title('主成分分析 (PCA) 次元削減 - I最適化', fontsize=16, weight='bold')
        else:
            plt.title('主成分分析 (PCA) 次元削減', fontsize=16, weight='bold')
            
        plt.xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # ES: Guardar PCA con sufijo según el tipo de optimización | EN: Save PCA with suffix based on optimization type | JA: 最適化タイプに応じた接尾辞でPCAを保存
        if optimization_type in ["d", "D", "d_optimal"]:
            pca_path = os.path.join(output_folder, "reduccion_dimensionalidad_pca_D.png")
        elif optimization_type in ["i", "I", "i_optimal"]:
            pca_path = os.path.join(output_folder, "reduccion_dimensionalidad_pca_I.png")
        else:
            pca_path = os.path.join(output_folder, "reduccion_dimensionalidad_pca.png")
            
        plt.savefig(pca_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ PCA保存完了: {pca_path}")
        return [pca_path]

def visualize_umap_enhanced(X_scaled, d_indices, i_indices, existing_indices, variable_names, output_folder, optimization_type="both", selected_d_df=None, selected_i_df=None):
    """ES: 📈 Visualización UMAP mejorada (mantiene compatibilidad)
    EN: 📈 Enhanced UMAP visualization (keeps backward compatibility)
    JA: 📈 改良版UMAP可視化（互換性維持）
    """
    # ES: Usar la nueva función separada | EN: Use the new separated function | JA: 新しい分離関数を使用
    return visualize_separate_dimension_reduction(X_scaled, d_indices, i_indices, existing_indices, variable_names, output_folder, optimization_type, selected_d_df, selected_i_df)

def get_project_name(sample_file):
    return os.path.splitext(os.path.basename(sample_file))[0]

def get_incremental_folder(base_dir, prefix):
    today = datetime.now().strftime('%Y%m%d')
    i = 1
    while True:
        folder = os.path.join(base_dir, f"{prefix}_{today}_{i:03d}")
        if not os.path.exists(folder):
            return folder
        i += 1

def run_integrated_optimizer(sample_file, existing_data_file=None, output_folder=".", num_experiments=15, 
                           sample_size=None, enable_hyperparameter_tuning=True, force_reoptimization=False, optimization_type="both"):
    """
    ES: Ejecuta el optimizador integrado D-óptimo + I-óptimo.
    EN: Run the integrated D-optimal + I-optimal optimizer.
    JA: D最適 + I最適 の統合オプティマイザを実行。

    ES: Parámetros:
    EN: Parameters:
    JA: 引数:

    - sample_file:
      ES: Excel con combinaciones de muestras (sample_combinations.xlsx)
      EN: Excel file with sample combinations (sample_combinations.xlsx)
      JA: サンプル組合せのExcel（sample_combinations.xlsx）
    - existing_data_file:
      ES: Excel con datos experimentales existentes (opcional)
      EN: Excel file with existing experimental data (optional)
      JA: 既存実験データのExcel（任意）
    - output_folder:
      ES: Carpeta de salida de resultados
      EN: Results output folder
      JA: 結果出力フォルダ
    - num_experiments:
      ES: Número de experimentos a seleccionar
      EN: Number of experiments to select
      JA: 選択実験数
    - sample_size:
      ES: Tamaño de muestreo para reducción (opcional)
      EN: Sample size for candidate reduction (optional)
      JA: 削減用サンプルサイズ（任意）
    - enable_hyperparameter_tuning:
      ES: Habilitar optimización de hiperparámetros de UMAP
      EN: Enable UMAP hyperparameter tuning
      JA: UMAPハイパーパラメータ最適化を有効化
    - force_reoptimization:
      ES: Forzar re-optimización de hiperparámetros de UMAP
      EN: Force UMAP hyperparameter re-optimization
      JA: UMAPハイパーパラメータ再最適化を強制
    - optimization_type:
      ES: "d", "i" o "both" (tipo de optimización)
      EN: "d", "i", or "both" (optimization type)
      JA: "d" / "i" / "both"（最適化タイプ）
    """
    print("🚀 化学実験計画システム - 統合バージョン")
    print("="*60)
    if optimization_type in ["d", "D", "d_optimal"]:
        print("📊 D最適化専用グラフ生成")
    elif optimization_type in ["i", "I", "i_optimal"]:
        print("📊 I最適化専用グラフ生成")
    else:
        print("📊 特徴量ヒストグラム（色分け）")
        print("📈 次元削減UMAPの可視化（強化版）")
    print("="*60)

    # ES: Crear carpeta de salida directamente en output_folder
    # EN: Create the output folder directly under output_folder
    # JA: output_folder 直下に出力フォルダを作成
    project_name = get_project_name(sample_file)
    di_folder = output_folder  # Use output_folder directly (no intermediate folder)
    os.makedirs(di_folder, exist_ok=True)

    # ES: Leer archivo de combinaciones de muestras
    # EN: Read sample combination file
    # JA: サンプル組合せファイルを読み込み
    print(f"\n📊 サンプル組合せファイルを読み込み中...")
    sample_ext = os.path.splitext(str(sample_file))[1].lower()
    full_df = pd.read_csv(sample_file, encoding="utf-8-sig") if sample_ext == ".csv" else pd.read_excel(sample_file)

    # ES: Usar SOLO 7 variables core para optimización/visualización (no incluir ブラシ one-hot ni 線材長)
    # EN: Use only 7 core variables for optimization/visualization (exclude ブラシ one-hot and 線材長)
    # JA: 最適化/可視化にはコア7変数のみ使用（ブラシ one-hot・線材長は含めない）
    dir_col = "UPカット" if "UPカット" in full_df.columns else ("回転方向" if "回転方向" in full_df.columns else None)
    if dir_col is None:
        raise ValueError("❌ Falta columna de dirección: 'UPカット' o '回転方向'")
    design_cols = ["回転速度", "送り速度", dir_col, "切込量", "突出量", "載せ率", "パス数"]
    missing = [c for c in design_cols if c not in full_df.columns]
    if missing:
        raise ValueError(f"❌ Faltan columnas de diseño: {missing}")

    candidate_df = full_df[design_cols].copy()
    candidate_points = candidate_df.values
    variable_names = design_cols
    
    print(f"✅ サンプル組合せファイル読み込み完了:")
    print(f"  - 説明変数数: {len(variable_names)}")
    print(f"  - 候補点数: {len(candidate_points):,}")
    print(f"  - 説明変数: {variable_names}")

    # ES: Procesar datos experimentales existentes
    # EN: Process existing experimental data
    # JA: 既存実験データを処理
    existing_indices = []
    if existing_data_file and os.path.exists(existing_data_file):
        print(f"\n🔍 既存実験データ処理中...")
        
        # ES: Crear DataFrame de compatibilidad temporal para diseño
        # EN: Create a temporary compatibility DataFrame for the design table
        # JA: 設計表との一時互換DataFrameを作成
        design_df = pd.DataFrame({
            "説明変数名": variable_names,
            "最小値": [candidate_df[var].min() for var in variable_names],
            "最大値": [candidate_df[var].max() for var in variable_names],
            "刻み幅": [1.0] * len(variable_names)  # デフォルト値
        })
        
        existing_data, available_vars = load_and_validate_existing_data(
            existing_data_file, design_df, verbose=True
        )

        if existing_data is not None and len(existing_data) > 0:
            existing_indices = match_existing_experiments_enhanced(
                candidate_points, existing_data, variable_names,
                tolerance_relative=1e-4,
                tolerance_absolute=1e-6,
                verbose=True
            )
        else:
            print("❌ 既存実験データ利用不可")
    else:
        print("ℹ️ 既存実験ファイル指定なしまたは存在しない")

    # === 候補点削減 if excede umbral ===
    original_candidate_count = len(candidate_points)
    should_reduce = len(candidate_points) > CANDIDATE_REDUCTION_THRESHOLD

    if should_reduce:
        max_candidates = sample_size if sample_size else MAX_REDUCED_CANDIDATES
        print(f"\n🔄 候補点削減実行: {original_candidate_count:,} → {max_candidates:,}")
        candidate_points, reduced_mapping = hierarchical_candidate_reduction(
            candidate_points, max_candidates, existing_indices
        )

        if existing_indices:
            existing_indices = [reduced_mapping.index(idx) for idx in existing_indices if idx in reduced_mapping]
            print(f"✅ 既存実験点マッピング更新完了: {len(existing_indices)} 保持")

        # ES: Reducir también el DF completo para que índices coincidan
        # EN: Also reduce the full DF so indices match
        # JA: インデックスが一致するようフルDFも削減
        try:
            full_df = full_df.iloc[reduced_mapping].reset_index(drop=True)
        except Exception:
            pass
        candidate_df = pd.DataFrame(candidate_points, columns=variable_names)

    print(f"\n✅ 最終データセット:")
    print(f"  - 最終候補点数: {len(candidate_points):,}")
    print(f"  - 既存実験点数: {len(existing_indices)}")
    print(f"  - 既存実験点利用率: {len(existing_indices)/num_experiments*100:.1f}%")

    # === 前処理データ ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(candidate_points)
    print(f"✅ データ標準化完了")

    # === D最適計画 ===
    print(f"\n🎯 D最適計画を実行中")
    d_indices, d_score = select_d_optimal_design_enhanced(
        X_scaled, existing_indices, num_experiments, verbose=VERBOSE
    )
    print(f"✅ D最適計画完了")
    print(f"  - 最終スコア: {d_score:.4f}")
    print(f"  - 選択点数: {len(d_indices)}")
    print(f"  - 既存点数: {len([i for i in d_indices if i in existing_indices])} 点")
    print(f"  - 新規点数: {len([i for i in d_indices if i not in existing_indices])} 点")

    # === I最適計画 ===
    print(f"\n🎯 I最適計画を実行中")
    i_indices = select_i_optimal_design(
        X_scaled, num_experiments, existing_indices
    )
    print(f"✅ I最適計画完了")
    print(f"  - 選択点数: {len(i_indices)}")
    print(f"  - 既存点数: {len([i for i in i_indices if i in existing_indices])} 点")
    print(f"  - 新規点数: {len([i for i in i_indices if i not in existing_indices])} 点")

    # === 結果処理 (新規点のみ) ===
    d_new_indices = [idx for idx in d_indices if idx not in existing_indices]
    i_new_indices = [idx for idx in i_indices if idx not in existing_indices]

    # Exportar DF completo (incluye A13/A11/A21/A32, 線材長, etc.), pero optimizar con candidate_df (solo core)
    selected_d_df = full_df.iloc[d_new_indices].copy() if d_new_indices else pd.DataFrame()
    selected_i_df = full_df.iloc[i_new_indices].copy() if i_new_indices else pd.DataFrame()

    print(f"\n📊 選択結果サマリー:")
    print(f"  - 既存実験点利用: {len(existing_indices)} 点")
    print(f"  - D最適新規選択: {len(d_new_indices)} 点")
    print(f"  - I最適新規選択: {len(i_new_indices)} 点")
    print(f"  - D最適全体: {len(d_indices)} 点")
    print(f"  - I最適全体: {len(i_indices)} 点")

    # NOTE: mantener nombres originales (面粗度(Ra)前/後) para compatibilidad con reconocimiento/export en GUI

    # === 後処理ファイルパス準備 ===
    d_path = os.path.join(di_folder, "D_optimal_新規実験点.xlsx")
    i_path = os.path.join(di_folder, "I最適化_新規実験点.xlsx")
    all_d_path = os.path.join(di_folder, "D最適化_全実験点.xlsx")
    all_i_path = os.path.join(di_folder, "I最適化_全実験点.xlsx")
    candidate_path = os.path.join(di_folder, "候補点一覧_v2.xlsx")

    # === 可視化 ===
    print(f"\n📊 特徴量分布の可視化開始...")
    
    # 特徴量ヒストグラム (1変数ごと) - 最適化タイプを指定
    hist_paths = visualize_feature_histograms(candidate_df, d_indices, i_indices, existing_indices, variable_names, di_folder, optimization_type)
    
    # 次元削減UMAP - 最適化タイプを指定
    umap_path = visualize_umap_enhanced(X_scaled, d_indices, i_indices, existing_indices, variable_names, di_folder, optimization_type)

    print(f"\n🎉 化学実験計画システム（統合版）完了")
    print("="*60)
    if optimization_type in ["d", "D", "d_optimal"]:
        print("✅ D最適化専用グラフ生成完了")
    elif optimization_type in ["i", "I", "i_optimal"]:
        print("✅ I最適化専用グラフ生成完了")
    else:
        print("✅ 既存実験点を活用した最適実験計画完了")
        print("📊 可視化: 特徴量分布ヒストグラム + 次元削減UMAP")
    print("💾 ExcelファイルはOKボタンを押した時に保存されます")
    print("="*60)

    # ES: Añadir D基準値 solo si d_score está definido | EN: Add D基準値 only if d_score is defined | JA: d_score が定義されている場合のみ D基準値 を追加
    if not selected_d_df.empty and 'd_score' in locals():
        selected_d_df['No.'] = range(1, len(selected_d_df) + 1)
        if 'パス数' in selected_d_df.columns:
            insert_at = selected_d_df.columns.get_loc('パス数') + 1
        else:
            insert_at = len(selected_d_df.columns)
        selected_d_df.insert(insert_at, 'D基準値', d_score)
        cols = ['No.'] + [c for c in selected_d_df.columns if c != 'No.']
        selected_d_df = selected_d_df[cols]
    # ES: Añadir I基準値 (placeholder) | EN: Add I基準値 (placeholder) | JA: I基準値 を追加（プレースホルダ）
    if not selected_i_df.empty:
        selected_i_df['No.'] = range(1, len(selected_i_df) + 1)
        if 'パス数' in selected_i_df.columns:
            insert_at = selected_i_df.columns.get_loc('パス数') + 1
        else:
            insert_at = len(selected_i_df.columns)
        selected_i_df.insert(insert_at, 'I基準値', '')  # Placeholder value
        cols = ['No.'] + [c for c in selected_i_df.columns if c != 'No.']
        selected_i_df = selected_i_df[cols]

    # ヒストグラムをdi_folderに保存
    hist_paths = visualize_feature_histograms(candidate_df, d_indices, i_indices, existing_indices, variable_names, di_folder, optimization_type)
    # 次元削減グラフを個別に保存 (PCA + UMAP) - サンプル番号付き
    dimension_paths = visualize_separate_dimension_reduction(X_scaled, d_indices, i_indices, existing_indices, variable_names, di_folder, optimization_type, selected_d_df, selected_i_df)
    
    # NO 最適化中のExcel保存, ルートのみ準備
    # ExcelファイルはOKボタンを押した時に保存されます
    return {
        "d_dataframe": selected_d_df,
        "i_dataframe": selected_i_df,
        "d_path": d_path,
        "i_path": i_path,
        "all_d_path": all_d_path,
        "all_i_path": all_i_path,
        "candidate_path": candidate_path,
        "image_paths": hist_paths + dimension_paths,
        "d_indices": d_indices,
        "i_indices": i_indices,
        "existing_indices": existing_indices,
        "candidate_df": candidate_df,  # 保存候補点リストが必要な場合
        "all_d_df": candidate_df.iloc[d_indices].copy() if len(d_indices) > 0 else pd.DataFrame(),
        "all_i_df": candidate_df.iloc[i_indices].copy() if len(i_indices) > 0 else pd.DataFrame(),
        "output_folders": {"images": di_folder},
    }

# ES: Nueva función: guardar PCA y UMAP por separado, con etiquetas de muestra. Referencia: D_and_I最適化_Greedy法_ver3.py
# EN: New function: save PCA and UMAP separately, with sample labels. Reference: D_and_I最適化_Greedy法_ver3.py
# JA: 新関数: PCAとUMAPを個別に保存、サンプルラベル付き
# ES: En el archivo D_and_I最適化_Greedy法_ver3.py, sí se calcula tanto el I基準値 (I-criterion) como el D基準値 (D-criterion).
# EN: In D_and_I最適化_Greedy法_ver3.py, both I基準値 (I-criterion) and D基準値 (D-criterion) are computed.
# JA: D_and_I最適化_Greedy法_ver3.py では I基準値 と D基準値 の両方を計算しています。
# ES: Normalmente, el cálculo del D基準値 se realiza usando el determinante del submatriz de diseño seleccionada (...)
# EN: Typically, D基準値 is computed from the determinant of the selected design submatrix (...)
# JA: 通常、D基準値 は選択設計行列の部分行列の行列式などから計算します（例: log(det(XᵀX))）。
# ES: y el I基準値 se calcula como la mínima distancia entre puntos seleccionados (...)
# EN: and I基準値 is computed as the minimum distance between selected points (...)
# JA: I基準値 は選択点間の最小距離などで計算します（例: cdist と最小値）。
# ES: Busca funciones o bloques de código con nombres como \"calculate_d_criterion\", \"calculate_i_criterion\" (...)
# EN: Look for code blocks named \"calculate_d_criterion\" / \"calculate_i_criterion\" or using np.linalg.det / np.linalg.qr / cdist.
# JA: \"calculate_d_criterion\" / \"calculate_i_criterion\"、または np.linalg.det / np.linalg.qr / cdist を使う箇所を探してください。
# ES: En la mayoría de implementaciones, ambos valores se calculan para cada subconjunto candidato y se almacenan o se usan para seleccionar el mejor conjunto.
# EN: In most implementations, both metrics are computed per candidate subset and stored/used to pick the best subset.
# JA: 多くの実装では、候補サブセットごとに両指標を計算し、保存/選択に利用します。