#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[6]:


"""
化学実験計画法 - 既存実験点採用問題修正版
修正内容: 既存データから説明変数のみを抽出して照合
"""

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
import optuna
import warnings
warnings.filterwarnings('ignore')

# === 設定項目 ===
SETTING_FILE = "実験計画設定テンプレート.xlsx"
SHEET_DESIGN = "実験計画_説明変数設定"
SHEET_INFO = "実験計画_基本情報"
USE_EXISTING_DATA = True
EXISTING_DATA_FILE = "既存実験データ.xlsx"
MAX_TRIALS = 100000
VERBOSE = True
FONT_NAME = "Meiryo"

# === 最適化設定 ===
USE_NUMERICAL_STABLE_METHOD = True
CANDIDATE_REDUCTION_THRESHOLD = 10000
MAX_REDUCED_CANDIDATES = 5000
ENABLE_HYPERPARAMETER_TUNING = True
HYPERPARAMETER_CACHE_FILE = "umap_optimal_params.json"
FORCE_REOPTIMIZATION = False

DEFAULT_UMAP_PARAMS = {"n_neighbors": 15, "min_dist": 0.1}
rcParams['font.family'] = FONT_NAME

def load_and_validate_existing_data(existing_file, design_df, verbose=True):
    """
    🔧 修正: 既存実験データの読み込みと説明変数抽出

    化学実験データの特徴:
    - 説明変数（プロセス条件）+ 目的変数（品質特性）の混在
    - DOE照合には説明変数のみが必要
    - 数値精度とスケールの考慮が重要
    """
    try:
        existing_df = pd.read_excel(existing_file)
        if verbose:
            print(f"📁 既存実験データ読み込み: {len(existing_df)}行 × {len(existing_df.columns)}列")
            print(f"📋 既存データ列名: {list(existing_df.columns)}")

        # 説明変数名を取得
        variable_names = design_df["説明変数名"].tolist()
        if verbose:
            print(f"🎯 対象説明変数: {variable_names}")

        # 既存データから説明変数列のみを抽出
        missing_vars = []
        available_vars = []

        for var in variable_names:
            if var in existing_df.columns:
                available_vars.append(var)
            else:
                missing_vars.append(var)

        if missing_vars:
            print(f"⚠️ 既存データに以下の説明変数が見つかりません: {missing_vars}")
            if len(available_vars) < len(variable_names) * 0.7:  # 70%未満の変数しかない場合
                print("❌ 既存データの変数不足（70%未満）- 既存実験点の使用をスキップ")
                return None, []
            else:
                print(f"✅ 利用可能変数（{len(available_vars)}/{len(variable_names)}）で継続")

        # 説明変数のみを抽出
        existing_explanatory = existing_df[available_vars]

        if verbose:
            print(f"✅ 説明変数抽出完了: {len(existing_explanatory)}行 × {len(available_vars)}列")
            print(f"📊 データサンプル（最初の3行）:")
            print(existing_explanatory.head(3))
            print(f"📈 データ統計:")
            print(existing_explanatory.describe())

        # 化学プロセス特有の品質チェック
        # 1. 欠損値チェック
        missing_count = existing_explanatory.isnull().sum().sum()
        if missing_count > 0:
            print(f"⚠️ 欠損値検出: {missing_count}個")
            existing_explanatory = existing_explanatory.dropna()
            print(f"🔧 欠損値除去後: {len(existing_explanatory)}行")

        # 2. 重複実験点チェック（化学実験では重要）
        duplicates = existing_explanatory.duplicated().sum()
        if duplicates > 0:
            print(f"⚠️ 重複実験点検出: {duplicates}個")
            existing_explanatory = existing_explanatory.drop_duplicates()
            print(f"🔧 重複除去後: {len(existing_explanatory)}行")

        # 3. 数値範囲チェック（プロセス条件の妥当性）
        for var in available_vars:
            var_info = design_df[design_df["説明変数名"] == var].iloc[0]
            min_val, max_val = var_info["最小値"], var_info["最大値"]

            out_of_range = (existing_explanatory[var] < min_val) | (existing_explanatory[var] > max_val)
            out_count = out_of_range.sum()

            if out_count > 0:
                print(f"⚠️ {var}: 範囲外データ {out_count}個 (設定範囲: {min_val}～{max_val})")
                # 範囲外データも保持（実際の実験条件として有効な場合があるため）

        return existing_explanatory, available_vars

    except FileNotFoundError:
        print(f"❌ 既存実験データファイルが見つかりません: {existing_file}")
        return None, []
    except Exception as e:
        print(f"❌ 既存データ読み込みエラー: {e}")
        return None, []

def match_existing_experiments_enhanced(candidate_points, existing_data, variable_names, 
                                      tolerance_relative=1e-6, tolerance_absolute=1e-8, verbose=True):
    """
    🔧 修正: 化学実験条件の高精度マッチング

    化学プロセスの特徴を考慮:
    - 測定精度の違い（回転速度 vs 載せ率）
    - 相対誤差と絶対誤差の併用
    - スケール正規化による公平な比較
    """
    if existing_data is None or len(existing_data) == 0:
        return []

    print(f"🔍 既存実験点とのマッチング開始")
    print(f"  - 候補点数: {len(candidate_points):,}")
    print(f"  - 既存実験数: {len(existing_data)}")
    print(f"  - 許容誤差（相対）: {tolerance_relative}")
    print(f"  - 許容誤差（絶対）: {tolerance_absolute}")

    # 候補点をDataFrameに変換（列名統一）
    candidate_df = pd.DataFrame(candidate_points, columns=variable_names)

    # 両データセットを標準化（スケールの違いを吸収）
    scaler = StandardScaler()
    candidate_scaled = scaler.fit_transform(candidate_df)

    # 既存データも同じ変数順序で並び替え
    existing_aligned = existing_data[variable_names]
    existing_scaled = scaler.transform(existing_aligned)

    matched_indices = []
    match_details = []

    # 各既存実験点について最も近い候補点を探索
    for exist_idx, exist_row in enumerate(existing_aligned.values):
        min_distance = float('inf')
        best_candidate_idx = None

        for cand_idx, cand_row in enumerate(candidate_df.values):
            # 1. 相対誤差ベースの比較
            relative_errors = []
            absolute_ok = True

            for var_idx, var_name in enumerate(variable_names):
                exist_val = exist_row[var_idx]
                cand_val = cand_row[var_idx]

                # 絶対誤差チェック
                abs_error = abs(exist_val - cand_val)
                if abs_error > tolerance_absolute:
                    # 相対誤差もチェック
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
                # 総合距離（標準化空間での距離）
                distance = np.linalg.norm(existing_scaled[exist_idx] - candidate_scaled[cand_idx])

                if distance < min_distance:
                    min_distance = distance
                    best_candidate_idx = cand_idx

        if best_candidate_idx is not None:
            matched_indices.append(best_candidate_idx)

            # マッチング詳細を記録
            match_detail = {
                '既存実験番号': exist_idx,
                '候補点番号': best_candidate_idx,
                '距離': min_distance,
                '既存実験条件': existing_aligned.iloc[exist_idx].to_dict(),
                '候補点条件': candidate_df.iloc[best_candidate_idx].to_dict()
            }
            match_details.append(match_detail)

            if verbose and len(matched_indices) <= 5:  # 最初の5件を詳細表示
                print(f"✅ マッチング {len(matched_indices)}: 既存#{exist_idx} → 候補#{best_candidate_idx} (距離: {min_distance:.4f})")

    # 重複除去（1つの候補点に複数の既存実験がマッチした場合）
    unique_matched = list(set(matched_indices))

    print(f"📊 マッチング結果:")
    print(f"  - 初期マッチ数: {len(matched_indices)}")
    print(f"  - 重複除去後: {len(unique_matched)}")
    print(f"  - マッチング率: {len(unique_matched)/len(existing_data)*100:.1f}%")

    if len(unique_matched) == 0:
        print("⚠️ 既存実験点がマッチしませんでした")
        print("💡 考えられる原因:")
        print("  1. 既存実験条件が候補点の設定範囲外")
        print("  2. 刻み幅設定が既存データと合わない")
        print("  3. 許容誤差設定が厳しすぎる")

        # 診断情報の提供
        print("\n🔍 診断情報:")
        for var in variable_names:
            exist_range = (existing_aligned[var].min(), existing_aligned[var].max())
            cand_range = (candidate_df[var].min(), candidate_df[var].max())
            print(f"  {var}: 既存{exist_range} vs 候補{cand_range}")

    return unique_matched

def hierarchical_candidate_reduction(candidate_points, max_candidates=5000, existing_indices=None):
    """階層的サンプリングによる候補点削減（既存点保持）"""
    n_original = len(candidate_points)

    if n_original <= max_candidates:
        print(f"📊 候補点数({n_original:,})は削減不要（閾値: {max_candidates:,}）")
        return candidate_points, list(range(n_original))

    print(f"🔄 ✅ 階層的サンプリング実行: {n_original:,} → {max_candidates:,}点に削減")

    # 既存実験点を保護
    if existing_indices:
        existing_set = set(existing_indices)
        available_indices = [i for i in range(n_original) if i not in existing_set]
        available_points = candidate_points[available_indices]
        n_to_select = max_candidates - len(existing_indices)
        print(f"📍 既存実験点保持: {len(existing_indices)}点")
    else:
        available_indices = list(range(n_original))
        available_points = candidate_points
        n_to_select = max_candidates
        existing_indices = []

    if n_to_select <= 0:
        print("⚠️ 既存点のみで上限に達しました")
        return candidate_points[existing_indices], existing_indices

    print(f"🎯 新規選定対象: {n_to_select:,}点")

    try:
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = min(n_to_select, len(available_points))
        print(f"🔧 MiniBatchKMeansクラスタリング: {n_clusters}クラスター")

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            batch_size=min(1000, len(available_points)//10),
            n_init=3,
            max_iter=100
        )

        start_time = time.time()
        clusters = kmeans.fit_predict(available_points)
        clustering_time = time.time() - start_time
        print(f"⏱️ クラスタリング時間: {clustering_time:.2f}秒")

        # 各クラスターから代表点を選択
        selected_indices = list(existing_indices)  # 既存点は必ず保持

        for i in range(n_clusters):
            cluster_mask = clusters == i
            if np.any(cluster_mask):
                cluster_indices_in_available = np.where(cluster_mask)[0]
                cluster_original_indices = [available_indices[j] for j in cluster_indices_in_available]

                # クラスター重心に最も近い点を選択
                cluster_points = available_points[cluster_mask]
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_points - center, axis=1)
                closest_idx_in_cluster = np.argmin(distances)
                closest_original_idx = cluster_original_indices[closest_idx_in_cluster]

                selected_indices.append(closest_original_idx)

        reduced_points = candidate_points[selected_indices]

        print(f"✅ 階層的サンプリング完了: 最終候補点数 {len(reduced_points):,}")
        print(f"  - 既存実験点保持: {len(existing_indices)}点")
        print(f"  - 新規選定点: {len(selected_indices) - len(existing_indices)}点")

        return reduced_points, selected_indices

    except Exception as e:
        print(f"⚠️ 階層的サンプリングエラー: {e}")
        # フォールバック処理
        return candidate_points, list(range(len(candidate_points)))

def calculate_d_criterion_stable(X, method='auto'):
    """数値的に安定なD-criterion計算"""
    try:
        condition_number = np.linalg.cond(X)

        if USE_NUMERICAL_STABLE_METHOD or method == 'auto' and condition_number > 1e12:
            method = 'svd'
            if VERBOSE and condition_number > 1e12:
                print(f"🔧 高条件数検出({condition_number:.2e}) - SVD法適用")

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
            print(f"⚠️ D-criterion計算エラー: {e}")
        return -np.inf, np.inf

def select_d_optimal_design_enhanced(X_all, existing_indices, new_experiments, verbose=True):
    """
    D-optimal設計選定（既存実験点 + 新規実験点）

    Args:
        X_all: 全候補点
        existing_indices: 既存実験点のインデックス
        new_experiments: 新規実験点数
        verbose: 詳細表示
    """
    base = list(existing_indices) if existing_indices else []
    remaining = [i for i in range(len(X_all)) if i not in base]
    total_select = len(base) + new_experiments  # 既存 + 新規

    if verbose:
        print(f"  - 既存実験点: {len(base)}点")
        print(f"  - 新規実験点: {new_experiments}点")
        print(f"  - 合計選定点: {total_select}点")

    if new_experiments <= 0:
        if verbose:
            print(f"  ✅ 既存実験点のみで完了")
        score, _ = calculate_d_criterion_stable(X_all[base])
        return base, score

    selected = list(base)

    for step in range(new_experiments):
        best_candidate = None
        best_score = -np.inf

        # 大規模データではサンプリング
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
                print(f"  ✅ 新規選定 {step+1}/{new_experiments}: 点{best_candidate}, スコア: {best_score:.4f}")
        else:
            if verbose:
                print(f"  ⚠️ ステップ{step+1}で適切な候補点が見つかりませんでした")
            break

    final_score, final_condition = calculate_d_criterion_stable(X_all[selected])
    return selected, final_score

    selected = list(base)

    for step in range(n_additional):
        best_candidate = None
        best_score = -np.inf

        # 大規模データではサンプリング
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
                print(f"✅ D-optimal選定 {step+1}/{n_additional}: 点{best_candidate}, スコア: {best_score:.4f}")
        else:
            if verbose:
                print(f"⚠️ ステップ{step+1}で適切な候補点が見つかりませんでした")
            break

    final_score, final_condition = calculate_d_criterion_stable(X_all[selected])
    return selected, final_score

def select_i_optimal_design(X_all, new_experiments, existing_indices=None):
    """
    I-optimal設計選定（既存実験点 + 新規実験点）

    Args:
        X_all: 全候補点
        new_experiments: 新規実験点数
        existing_indices: 既存実験点のインデックス
    """
    if existing_indices:
        selected_indices = list(existing_indices)
        print(f"  - 既存実験点: {len(existing_indices)}点")
        print(f"  - 新規実験点: {new_experiments}点")
        print(f"  - 合計選定点: {len(existing_indices) + new_experiments}点")
    else:
        selected_indices = [0]
        print(f"  - 新規実験点: {new_experiments}点（既存点なし）")

    remaining_indices = [i for i in range(len(X_all)) if i not in selected_indices]
    target_total = len(selected_indices) + new_experiments

    step = 0
    while len(selected_indices) < target_total and remaining_indices:
        dists = cdist(X_all[remaining_indices], X_all[selected_indices])
        min_dists = dists.min(axis=1)
        next_idx_in_remaining = np.argmax(min_dists)
        next_index = remaining_indices[next_idx_in_remaining]
        selected_indices.append(next_index)
        remaining_indices.remove(next_index)
        step += 1
        print(f"  ✅ 新規選定 {step}/{new_experiments}: 点{next_index}")

    return selected_indices

    while len(selected_indices) < total_select:
        dists = cdist(X_all[remaining_indices], X_all[selected_indices])
        min_dists = dists.min(axis=1)
        next_idx_in_remaining = np.argmax(min_dists)
        next_index = remaining_indices[next_idx_in_remaining]
        selected_indices.append(next_index)
        remaining_indices.remove(next_index)

    return selected_indices

def generate_candidate_points(design_df):
    """候補点生成"""
    levels = []
    for _, row in design_df.iterrows():
        levels.append(np.arange(row["最小値"], row["最大値"] + row["刻み幅"], row["刻み幅"]))
    return np.array(list(product(*levels)))

def main():
    """メイン実行関数（既存実験点対応修正版）"""
    print("🚀 化学実験計画法システム - 既存実験点対応修正版")
    print("="*60)
    print("🔧 修正内容: 既存データから説明変数のみを抽出して照合")
    print("="*60)

    # 設定読み込み
    try:
        design_df = pd.read_excel(SETTING_FILE, sheet_name=SHEET_DESIGN)
        info_df = pd.read_excel(SETTING_FILE, sheet_name=SHEET_INFO)
        n_experiments = int(info_df.loc[info_df["設定項目"] == "実験数", "値"].values[0])
        print(f"📋 設定読み込み完了")
        print(f"  - 説明変数数: {len(design_df)}")
        print(f"  - 目標実験数: {n_experiments}")
    except Exception as e:
        print(f"❌ 設定読み込みエラー: {e}")
        return

    # 候補点生成
    print(f"\n📊 候補点生成中...")
    candidate_points = generate_candidate_points(design_df)
    print(f"✅ 初期候補点生成完了: {len(candidate_points):,}点")

    # 🔧 修正: 既存実験データの適切な処理
    existing_indices = []
    if USE_EXISTING_DATA:
        print(f"\n🔍 既存実験データ処理開始")

        # 既存データの読み込みと検証
        existing_data, available_vars = load_and_validate_existing_data(
            EXISTING_DATA_FILE, design_df, verbose=True
        )

        if existing_data is not None and len(existing_data) > 0:
            # 高精度マッチング実行
            variable_names = design_df["説明変数名"].tolist()
            existing_indices = match_existing_experiments_enhanced(
                candidate_points, existing_data, variable_names,
                tolerance_relative=1e-4,  # 化学プロセス用に緩和
                tolerance_absolute=1e-6,  # 化学プロセス用に緩和
                verbose=True
            )
        else:
            print("❌ 既存実験データが利用できません")

    # 候補点削減（既存点を保持）
    original_candidate_count = len(candidate_points)
    should_reduce = len(candidate_points) > CANDIDATE_REDUCTION_THRESHOLD

    if should_reduce:
        print(f"\n🔄 候補点削減実行: {original_candidate_count:,} → {MAX_REDUCED_CANDIDATES:,}")
        candidate_points, reduced_mapping = hierarchical_candidate_reduction(
            candidate_points, MAX_REDUCED_CANDIDATES, existing_indices
        )

        # 既存実験点インデックスの更新
        if existing_indices:
            existing_indices = [reduced_mapping.index(idx) for idx in existing_indices if idx in reduced_mapping]
            print(f"✅ 既存実験点マッピング更新: {len(existing_indices)}件保持")

    print(f"\n✅ 最終データセット:")
    print(f"  - 最終候補点数: {len(candidate_points):,}")
    print(f"  - 既存実験点数: {len(existing_indices)}")
    print(f"  - 既存実験活用率: {len(existing_indices)/n_experiments*100:.1f}%")

    # データ前処理
    candidate_df = pd.DataFrame(candidate_points, columns=design_df["説明変数名"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(candidate_points)
    print(f"✅ データ標準化完了")

    # D-optimal設計
    print(f"\n🎯 D-optimal設計実行（既存{len(existing_indices)}点を含む）")
    d_indices, d_score = select_d_optimal_design_enhanced(
        X_scaled, existing_indices, n_experiments, verbose=VERBOSE
    )
    print(f"✅ D-optimal設計完了")
    print(f"  - 最終スコア: {d_score:.4f}")
    print(f"  - 選定点数: {len(d_indices)}")
    print(f"  - 既存点活用: {len([i for i in d_indices if i in existing_indices])}点")

    # I-optimal設計
    print(f"\n🎯 I-optimal設計実行（既存{len(existing_indices)}点を含む）")
    i_indices = select_i_optimal_design(X_scaled, n_experiments, existing_indices)
    print(f"✅ I-optimal設計完了")
    print(f"  - 選定点数: {len(i_indices)}")
    print(f"  - 既存点活用: {len([i for i in i_indices if i in existing_indices])}点")

    # 結果保存
    selected_d_df = candidate_df.iloc[d_indices]
    selected_i_df = candidate_df.iloc[i_indices]

    # 既存/新規の区別を追加
    selected_d_df['データ種別'] = ['既存' if i in existing_indices else '新規' for i in d_indices]
    selected_i_df['データ種別'] = ['既存' if i in existing_indices else '新規' for i in i_indices]

    selected_d_df.to_excel("D_optimal_修正版.xlsx", index=False)
    selected_i_df.to_excel("I_optimal_修正版.xlsx", index=False)

    print(f"\n📊 最終結果サマリー:")
    print(f"  - D-optimal: 既存{len([i for i in d_indices if i in existing_indices])}点 + 新規{len(d_indices) - len([i for i in d_indices if i in existing_indices])}点")
    print(f"  - I-optimal: 既存{len([i for i in i_indices if i in existing_indices])}点 + 新規{len(i_indices) - len([i for i in i_indices if i in existing_indices])}点")
    print(f"💾 結果ファイル保存: D_optimal_修正版.xlsx, I_optimal_修正版.xlsx")

    print(f"\n🎉 化学実験計画法システム完了（既存実験点対応修正版）")
    print("="*60)

if __name__ == "__main__":
    main()


# In[7]:


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
import optuna
import warnings
warnings.filterwarnings('ignore')

# === 設定項目 ===
SETTING_FILE = "実験計画設定テンプレート.xlsx"
SHEET_DESIGN = "実験計画_説明変数設定"
SHEET_INFO = "実験計画_基本情報"
USE_EXISTING_DATA = True
EXISTING_DATA_FILE = "既存実験データ.xlsx"
MAX_TRIALS = 100000
VERBOSE = True
FONT_NAME = "Meiryo"

# === 最適化設定 ===
USE_NUMERICAL_STABLE_METHOD = True
CANDIDATE_REDUCTION_THRESHOLD = 10000
MAX_REDUCED_CANDIDATES = 5000
ENABLE_HYPERPARAMETER_TUNING = True
HYPERPARAMETER_CACHE_FILE = "umap_optimal_params.json"
FORCE_REOPTIMIZATION = False

DEFAULT_UMAP_PARAMS = {"n_neighbors": 15, "min_dist": 0.1}
rcParams['font.family'] = FONT_NAME

def load_and_validate_existing_data(existing_file, design_df, verbose=True):
    """既存実験データの読み込みと説明変数抽出"""
    try:
        existing_df = pd.read_excel(existing_file)
        if verbose:
            print(f"📁 既存実験データ読み込み: {len(existing_df)}行 × {len(existing_df.columns)}列")
            print(f"📋 既存データ列名: {list(existing_df.columns)}")

        # 説明変数名を取得
        variable_names = design_df["説明変数名"].tolist()
        if verbose:
            print(f"🎯 対象説明変数: {variable_names}")

        # 既存データから説明変数列のみを抽出
        missing_vars = []
        available_vars = []

        for var in variable_names:
            if var in existing_df.columns:
                available_vars.append(var)
            else:
                missing_vars.append(var)

        if missing_vars:
            print(f"⚠️ 既存データに以下の説明変数が見つかりません: {missing_vars}")
            if len(available_vars) < len(variable_names) * 0.7:
                print("❌ 既存データの変数不足（70%未満）- 既存実験点の使用をスキップ")
                return None, []
            else:
                print(f"✅ 利用可能変数（{len(available_vars)}/{len(variable_names)}）で継続")

        # 説明変数のみを抽出
        existing_explanatory = existing_df[available_vars]

        if verbose:
            print(f"✅ 説明変数抽出完了: {len(existing_explanatory)}行 × {len(available_vars)}列")
            print(f"📊 データサンプル（最初の3行）:")
            print(existing_explanatory.head(3))
            print(f"📈 データ統計:")
            print(existing_explanatory.describe())

        # 化学プロセス特有の品質チェック
        # 1. 欠損値チェック
        missing_count = existing_explanatory.isnull().sum().sum()
        if missing_count > 0:
            print(f"⚠️ 欠損値検出: {missing_count}個")
            existing_explanatory = existing_explanatory.dropna()
            print(f"🔧 欠損値除去後: {len(existing_explanatory)}行")

        # 2. 重複実験点チェック
        duplicates = existing_explanatory.duplicated().sum()
        if duplicates > 0:
            print(f"⚠️ 重複実験点検出: {duplicates}個")
            existing_explanatory = existing_explanatory.drop_duplicates()
            print(f"🔧 重複除去後: {len(existing_explanatory)}行")

        # 3. 数値範囲チェック
        for var in available_vars:
            var_info = design_df[design_df["説明変数名"] == var].iloc[0]
            min_val, max_val = var_info["最小値"], var_info["最大値"]

            out_of_range = (existing_explanatory[var] < min_val) | (existing_explanatory[var] > max_val)
            out_count = out_of_range.sum()

            if out_count > 0:
                print(f"⚠️ {var}: 範囲外データ {out_count}個 (設定範囲: {min_val}～{max_val})")

        return existing_explanatory, available_vars

    except FileNotFoundError:
        print(f"❌ 既存実験データファイルが見つかりません: {existing_file}")
        return None, []
    except Exception as e:
        print(f"❌ 既存データ読み込みエラー: {e}")
        return None, []

def match_existing_experiments_enhanced(candidate_points, existing_data, variable_names, 
                                      tolerance_relative=1e-6, tolerance_absolute=1e-8, verbose=True):
    """化学実験条件の高精度マッチング"""
    if existing_data is None or len(existing_data) == 0:
        return []

    print(f"🔍 既存実験点とのマッチング開始")
    print(f"  - 候補点数: {len(candidate_points):,}")
    print(f"  - 既存実験数: {len(existing_data)}")
    print(f"  - 許容誤差（相対）: {tolerance_relative}")
    print(f"  - 許容誤差（絶対）: {tolerance_absolute}")

    # 候補点をDataFrameに変換
    candidate_df = pd.DataFrame(candidate_points, columns=variable_names)

    # 両データセットを標準化
    scaler = StandardScaler()
    candidate_scaled = scaler.fit_transform(candidate_df)

    # 既存データも同じ変数順序で並び替え
    existing_aligned = existing_data[variable_names]
    existing_scaled = scaler.transform(existing_aligned)

    matched_indices = []
    match_details = []

    # 各既存実験点について最も近い候補点を探索
    for exist_idx, exist_row in enumerate(existing_aligned.values):
        min_distance = float('inf')
        best_candidate_idx = None

        for cand_idx, cand_row in enumerate(candidate_df.values):
            # 相対誤差ベースの比較
            relative_errors = []
            absolute_ok = True

            for var_idx, var_name in enumerate(variable_names):
                exist_val = exist_row[var_idx]
                cand_val = cand_row[var_idx]

                # 絶対誤差チェック
                abs_error = abs(exist_val - cand_val)
                if abs_error > tolerance_absolute:
                    # 相対誤差もチェック
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
                # 総合距離（標準化空間での距離）
                distance = np.linalg.norm(existing_scaled[exist_idx] - candidate_scaled[cand_idx])

                if distance < min_distance:
                    min_distance = distance
                    best_candidate_idx = cand_idx

        if best_candidate_idx is not None:
            matched_indices.append(best_candidate_idx)

            # マッチング詳細を記録
            match_detail = {
                '既存実験番号': exist_idx,
                '候補点番号': best_candidate_idx,
                '距離': min_distance,
                '既存実験条件': existing_aligned.iloc[exist_idx].to_dict(),
                '候補点条件': candidate_df.iloc[best_candidate_idx].to_dict()
            }
            match_details.append(match_detail)

            if verbose and len(matched_indices) <= 5:
                print(f"✅ マッチング {len(matched_indices)}: 既存#{exist_idx} → 候補#{best_candidate_idx} (距離: {min_distance:.4f})")

    # 重複除去
    unique_matched = list(set(matched_indices))

    print(f"📊 マッチング結果:")
    print(f"  - 初期マッチ数: {len(matched_indices)}")
    print(f"  - 重複除去後: {len(unique_matched)}")
    print(f"  - マッチング率: {len(unique_matched)/len(existing_data)*100:.1f}%")

    if len(unique_matched) == 0:
        print("⚠️ 既存実験点がマッチしませんでした")
        print("💡 考えられる原因:")
        print("  1. 既存実験条件が候補点の設定範囲外")
        print("  2. 刻み幅設定が既存データと合わない")
        print("  3. 許容誤差設定が厳しすぎる")

        # 診断情報の提供
        print("\n🔍 診断情報:")
        for var in variable_names:
            exist_range = (existing_aligned[var].min(), existing_aligned[var].max())
            cand_range = (candidate_df[var].min(), candidate_df[var].max())
            print(f"  {var}: 既存{exist_range} vs 候補{cand_range}")

    return unique_matched

def hierarchical_candidate_reduction(candidate_points, max_candidates=5000, existing_indices=None):
    """階層的サンプリングによる候補点削減"""
    n_original = len(candidate_points)

    if n_original <= max_candidates:
        print(f"📊 候補点数({n_original:,})は削減不要（閾値: {max_candidates:,}）")
        return candidate_points, list(range(n_original))

    print(f"🔄 ✅ 階層的サンプリング実行: {n_original:,} → {max_candidates:,}点に削減")

    # 既存実験点を保護
    if existing_indices:
        existing_set = set(existing_indices)
        available_indices = [i for i in range(n_original) if i not in existing_set]
        available_points = candidate_points[available_indices]
        n_to_select = max_candidates - len(existing_indices)
        print(f"📍 既存実験点保持: {len(existing_indices)}点")
    else:
        available_indices = list(range(n_original))
        available_points = candidate_points
        n_to_select = max_candidates
        existing_indices = []

    if n_to_select <= 0:
        print("⚠️ 既存点のみで上限に達しました")
        return candidate_points[existing_indices], existing_indices

    print(f"🎯 新規選定対象: {n_to_select:,}点")

    try:
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = min(n_to_select, len(available_points))
        print(f"🔧 MiniBatchKMeansクラスタリング: {n_clusters}クラスター")

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            batch_size=min(1000, len(available_points)//10),
            n_init=3,
            max_iter=100
        )

        start_time = time.time()
        clusters = kmeans.fit_predict(available_points)
        clustering_time = time.time() - start_time
        print(f"⏱️ クラスタリング時間: {clustering_time:.2f}秒")

        # 各クラスターから代表点を選択
        selected_indices = list(existing_indices)

        for i in range(n_clusters):
            cluster_mask = clusters == i
            if np.any(cluster_mask):
                cluster_indices_in_available = np.where(cluster_mask)[0]
                cluster_original_indices = [available_indices[j] for j in cluster_indices_in_available]

                # クラスター重心に最も近い点を選択
                cluster_points = available_points[cluster_mask]
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_points - center, axis=1)
                closest_idx_in_cluster = np.argmin(distances)
                closest_original_idx = cluster_original_indices[closest_idx_in_cluster]

                selected_indices.append(closest_original_idx)

        reduced_points = candidate_points[selected_indices]

        print(f"✅ 階層的サンプリング完了: 最終候補点数 {len(reduced_points):,}")
        print(f"  - 既存実験点保持: {len(existing_indices)}点")
        print(f"  - 新規選定点: {len(selected_indices) - len(existing_indices)}点")

        return reduced_points, selected_indices

    except Exception as e:
        print(f"⚠️ 階層的サンプリングエラー: {e}")
        return candidate_points, list(range(len(candidate_points)))

def calculate_d_criterion_stable(X, method='auto'):
    """数値的に安定なD-criterion計算"""
    try:
        condition_number = np.linalg.cond(X)

        if USE_NUMERICAL_STABLE_METHOD or method == 'auto' and condition_number > 1e12:
            method = 'svd'
            if VERBOSE and condition_number > 1e12:
                print(f"🔧 高条件数検出({condition_number:.2e}) - SVD法適用")

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
            print(f"⚠️ D-criterion計算エラー: {e}")
        return -np.inf, np.inf

def select_d_optimal_design_enhanced(X_all, existing_indices, new_experiments, verbose=True):
    """D-optimal設計選定（既存実験点 + 新規実験点）"""
    base = list(existing_indices) if existing_indices else []
    remaining = [i for i in range(len(X_all)) if i not in base]
    total_select = len(base) + new_experiments

    if verbose:
        print(f"  - 既存実験点: {len(base)}点")
        print(f"  - 新規実験点: {new_experiments}点")
        print(f"  - 合計選定点: {total_select}点")

    if new_experiments <= 0:
        if verbose:
            print(f"  ✅ 既存実験点のみで完了")
        score, _ = calculate_d_criterion_stable(X_all[base])
        return base, score

    selected = list(base)

    for step in range(new_experiments):
        best_candidate = None
        best_score = -np.inf

        # 大規模データではサンプリング
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
                print(f"  ✅ 新規選定 {step+1}/{new_experiments}: 点{best_candidate}, スコア: {best_score:.4f}")
        else:
            if verbose:
                print(f"  ⚠️ ステップ{step+1}で適切な候補点が見つかりませんでした")
            break

    final_score, final_condition = calculate_d_criterion_stable(X_all[selected])
    return selected, final_score

def select_i_optimal_design(X_all, new_experiments, existing_indices=None):
    """I-optimal設計選定（既存実験点 + 新規実験点）"""
    if existing_indices:
        selected_indices = list(existing_indices)
        print(f"  - 既存実験点: {len(existing_indices)}点")
        print(f"  - 新規実験点: {new_experiments}点")
        print(f"  - 合計選定点: {len(existing_indices) + new_experiments}点")
    else:
        selected_indices = [0]
        print(f"  - 新規実験点: {new_experiments}点（既存点なし）")

    remaining_indices = [i for i in range(len(X_all)) if i not in selected_indices]
    target_total = len(selected_indices) + new_experiments

    step = 0
    while len(selected_indices) < target_total and remaining_indices:
        dists = cdist(X_all[remaining_indices], X_all[selected_indices])
        min_dists = dists.min(axis=1)
        next_idx_in_remaining = np.argmax(min_dists)
        next_index = remaining_indices[next_idx_in_remaining]
        selected_indices.append(next_index)
        remaining_indices.remove(next_index)
        step += 1
        print(f"  ✅ 新規選定 {step}/{new_experiments}: 点{next_index}")

    return selected_indices

def generate_candidate_points(design_df):
    """候補点生成"""
    levels = []
    for _, row in design_df.iterrows():
        levels.append(np.arange(row["最小値"], row["最大値"] + row["刻み幅"], row["刻み幅"]))
    return np.array(list(product(*levels)))

# =================== 📊 可視化機能強化 ===================

def save_hyperparameters(params, filepath=HYPERPARAMETER_CACHE_FILE):
    """ハイパーパラメーター保存"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        print(f"✅ ハイパーパラメーター保存: {filepath}")
    except Exception as e:
        print(f"⚠️ 保存エラー: {e}")

def load_hyperparameters(filepath=HYPERPARAMETER_CACHE_FILE):
    """ハイパーパラメーター読み込み"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                params = json.load(f)
            print(f"✅ 保存済みハイパーパラメーター読み込み: {filepath}")
            return params
        return None
    except Exception as e:
        print(f"⚠️ 読み込みエラー: {e}")
        return None

def umap_objective_function(trial, X_scaled, d_indices, i_indices, existing_indices):
    """UMAP最適化目的関数"""
    try:
        import umap

        n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
        min_dist = trial.suggest_float("min_dist", 0.0, 0.5)

        # 計算効率のため、データ数が多い場合はサンプリング
        if len(X_scaled) > 2000:
            sample_indices = np.random.choice(len(X_scaled), 2000, replace=False)
            X_sample = X_scaled[sample_indices]

            # ラベルも対応してサンプリング
            d_sample = [i for i, idx in enumerate(sample_indices) if idx in d_indices]
            i_sample = [i for i, idx in enumerate(sample_indices) if idx in i_indices]
            existing_sample = [i for i, idx in enumerate(sample_indices) if idx in existing_indices]
        else:
            X_sample = X_scaled
            d_sample = d_indices
            i_sample = i_indices
            existing_sample = existing_indices

        reducer = umap.UMAP(
            n_neighbors=n_neighbors, 
            min_dist=min_dist, 
            n_components=2,
            random_state=42,
            n_jobs=1
        )
        embedding = reducer.fit_transform(X_sample)

        # 評価指標の計算
        labels = np.zeros(len(X_sample))
        if d_sample:
            labels[d_sample] = 1
        if i_sample:
            labels[i_sample] = 2
        if existing_sample:
            labels[existing_sample] = 3

        # 分離度の計算
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        centroids = []
        for label in unique_labels:
            if np.any(labels == label):
                centroid = embedding[labels == label].mean(axis=0)
                centroids.append(centroid)

        centroids = np.array(centroids)
        separation_score = np.mean(cdist(centroids, centroids)[np.triu_indices_from(centroids, k=1)])

        # 凝集度の計算
        cohesion_scores = []
        for label in unique_labels:
            cluster_points = embedding[labels == label]
            if len(cluster_points) > 1:
                centroid = cluster_points.mean(axis=0)
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                cohesion_scores.append(1.0 / (1.0 + np.mean(distances)))

        cohesion_score = np.mean(cohesion_scores) if cohesion_scores else 0.0

        # 総合スコア
        total_score = 0.7 * separation_score + 0.3 * cohesion_score
        return total_score

    except Exception as e:
        return 0.0

def get_umap_params_optimized(X_scaled, d_indices, i_indices, existing_indices):
    """UMAP ハイパーパラメーター最適化（選択可能）"""
    # 保存済みパラメーターの確認
    cached_params = load_hyperparameters()

    if cached_params is not None and not FORCE_REOPTIMIZATION:
        print(f"📁 保存済みUMAPパラメーター使用: {cached_params}")
        print("💡 ハイパーパラメーター最適化をスキップ（既に最適化済み）")
        return cached_params

    elif ENABLE_HYPERPARAMETER_TUNING:
        if cached_params is None:
            print("🔍 初回実行: UMAPハイパーパラメーター最適化を開始...")
        else:
            print("🔄 強制再最適化: UMAPハイパーパラメーター最適化を実行...")

        print("⚙️ 最適化設定:")
        print(f"  - 試行回数: 20回（効率重視）")
        print(f"  - 評価指標: 分離度70% + 凝集度30%")
        print(f"  - サンプリング: 2000点上限（大規模データ対応）")

        start_time = time.time()

        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(
            lambda trial: umap_objective_function(trial, X_scaled, d_indices, i_indices, existing_indices), 
            n_trials=20,
            show_progress_bar=False
        )

        optimization_time = time.time() - start_time
        best_params = study.best_params

        print(f"✅ 最適化完了 - 実行時間: {optimization_time:.1f}秒")
        print(f"🎯 最適スコア: {study.best_value:.4f}")
        print(f"🔧 最適パラメーター: {best_params}")

        # 結果を保存
        save_hyperparameters(best_params)
        print("💾 最適パラメーターを保存（次回実行時に自動使用）")

        return best_params

    else:
        print("⚙️ ハイパーパラメーター最適化無効 - デフォルト値使用")
        print(f"📋 デフォルトUMAPパラメーター: {DEFAULT_UMAP_PARAMS}")
        return DEFAULT_UMAP_PARAMS

def visualize_feature_histograms(candidate_df, d_indices, i_indices, existing_indices, variable_names):
    """📊 修正1: 各特徴量のヒストグラム色分け表示"""
    print(f"\n📊 特徴量分布可視化開始")

    n_features = len(variable_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 4 * n_rows))

    for i, var_name in enumerate(variable_names):
        plt.subplot(n_rows, n_cols, i + 1)

        # 全候補点のヒストグラム（背景）
        plt.hist(candidate_df[var_name], bins=30, alpha=0.3, color='lightgray', 
                label=f'全候補点 ({len(candidate_df)})', density=True)

        # 既存実験点
        if existing_indices:
            existing_values = candidate_df.iloc[existing_indices][var_name]
            plt.hist(existing_values, bins=15, alpha=0.8, color='blue', 
                    label=f'既存実験点 ({len(existing_indices)})', density=True)

        # D-optimal新規点
        d_new_indices = [idx for idx in d_indices if idx not in existing_indices]
        if d_new_indices:
            d_values = candidate_df.iloc[d_new_indices][var_name]
            plt.hist(d_values, bins=10, alpha=0.8, color='red', 
                    label=f'D-optimal新規 ({len(d_new_indices)})', density=True)

        # I-optimal新規点
        i_new_indices = [idx for idx in i_indices if idx not in existing_indices]
        if i_new_indices:
            i_values = candidate_df.iloc[i_new_indices][var_name]
            plt.hist(i_values, bins=10, alpha=0.8, color='green', 
                    label=f'I-optimal新規 ({len(i_new_indices)})', density=True)

        plt.title(f'{var_name}の分布', fontsize=12, weight='bold')
        plt.xlabel(var_name)
        plt.ylabel('密度')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"✅ 特徴量分布可視化完了")

def visualize_umap_enhanced(X_scaled, d_indices, i_indices, existing_indices, variable_names):
    """📈 修正2: UMAP次元削減可視化（ハイパーパラメーター最適化付き）"""
    print(f"\n📈 UMAP次元削減可視化開始")

    try:
        import umap

        # ハイパーパラメーター最適化
        best_params = get_umap_params_optimized(X_scaled, d_indices, i_indices, existing_indices)

        # UMAP実行
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
        print(f"⏱️ UMAP実行時間: {umap_time:.2f}秒")

        # 可視化
        plt.figure(figsize=(14, 10))

        # PCAとUMAPの比較表示
        plt.subplot(2, 2, 1)
        # PCA可視化
        pca = PCA(n_components=2, random_state=42)
        reduced_pca = pca.fit_transform(X_scaled)

        # 全候補点（背景）
        plt.scatter(reduced_pca[:, 0], reduced_pca[:, 1], alpha=0.2, s=8, color='lightgray', label='候補点')

        # 既存実験点
        if existing_indices:
            existing_pca = reduced_pca[existing_indices]
            plt.scatter(existing_pca[:, 0], existing_pca[:, 1], 
                       s=120, color='blue', alpha=0.9, marker='o', 
                       edgecolors='navy', linewidth=2, zorder=10,
                       label=f'既存実験点 ({len(existing_indices)})')

        # D-optimal新規点
        d_new = [idx for idx in d_indices if idx not in existing_indices]
        if d_new:
            d_pca = reduced_pca[d_new]
            plt.scatter(d_pca[:, 0], d_pca[:, 1], 
                       s=100, marker='x', color='red', linewidth=3, 
                       zorder=8, label=f'D-optimal新規 ({len(d_new)})')

        # I-optimal新規点
        i_new = [idx for idx in i_indices if idx not in existing_indices]
        if i_new:
            i_pca = reduced_pca[i_new]
            plt.scatter(i_pca[:, 0], i_pca[:, 1], 
                       s=100, marker='^', color='green', 
                       zorder=8, label=f'I-optimal新規 ({len(i_new)})')

        plt.title('PCA次元削減', fontsize=14, weight='bold')
        plt.xlabel(f'第1主成分 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'第2主成分 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)

        # UMAP可視化
        plt.subplot(2, 2, 2)

        # 全候補点（背景）
        plt.scatter(reduced_umap[:, 0], reduced_umap[:, 1], alpha=0.2, s=8, color='lightgray', label='候補点')

        # 既存実験点
        if existing_indices:
            existing_umap = reduced_umap[existing_indices]
            plt.scatter(existing_umap[:, 0], existing_umap[:, 1], 
                       s=120, color='blue', alpha=0.9, marker='o', 
                       edgecolors='navy', linewidth=2, zorder=10,
                       label=f'既存実験点 ({len(existing_indices)})')

            # 既存実験点に番号表示（最初の10点まで）
            for i, (x, y) in enumerate(existing_umap[:min(10, len(existing_umap))]):
                plt.annotate(f'{i+1}', (x, y), xytext=(3, 3), 
                           textcoords='offset points', fontsize=8, 
                           color='darkblue', weight='bold', zorder=11)

        # D-optimal新規点
        if d_new:
            d_umap = reduced_umap[d_new]
            plt.scatter(d_umap[:, 0], d_umap[:, 1], 
                       s=100, marker='x', color='red', linewidth=3, 
                       zorder=8, label=f'D-optimal新規 ({len(d_new)})')

        # I-optimal新規点
        if i_new:
            i_umap = reduced_umap[i_new]
            plt.scatter(i_umap[:, 0], i_umap[:, 1], 
                       s=100, marker='^', color='green', 
                       zorder=8, label=f'I-optimal新規 ({len(i_new)})')

        plt.title(f'UMAP次元削減 (n_neighbors={best_params["n_neighbors"]}, min_dist={best_params["min_dist"]:.3f})', 
                 fontsize=14, weight='bold')
        plt.xlabel('UMAP次元1')
        plt.ylabel('UMAP次元2')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)

        # ハイパーパラメーター最適化情報表示
        plt.subplot(2, 1, 2)
        plt.axis('off')

        info_text = f"""
🔧 UMAP最適化パラメーター:
   • n_neighbors: {best_params["n_neighbors"]} (近傍点数)
   • min_dist: {best_params["min_dist"]:.3f} (最小距離)

📊 データセット統計:
   • 全候補点: {len(X_scaled):,}点
   • 既存実験点: {len(existing_indices)}点
   • D-optimal総点数: {len(d_indices)}点 (既存{len([i for i in d_indices if i in existing_indices])} + 新規{len(d_new)})
   • I-optimal総点数: {len(i_indices)}点 (既存{len([i for i in i_indices if i in existing_indices])} + 新規{len(i_new)})

⏱️ 処理時間:
   • UMAP実行: {umap_time:.2f}秒
        """

        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        plt.tight_layout()
        plt.show()

        print(f"✅ UMAP次元削減可視化完了")

    except ImportError:
        print("❌ UMAP未インストール - PCAのみ表示")
        # PCAフォールバック可視化
        pca = PCA(n_components=2, random_state=42)
        reduced_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_pca[:, 0], reduced_pca[:, 1], alpha=0.2, s=8, color='lightgray', label='候補点')

        if existing_indices:
            existing_pca = reduced_pca[existing_indices]
            plt.scatter(existing_pca[:, 0], existing_pca[:, 1], 
                       s=120, color='blue', alpha=0.9, marker='o', 
                       edgecolors='navy', linewidth=2, zorder=10,
                       label=f'既存実験点 ({len(existing_indices)})')

        d_new = [idx for idx in d_indices if idx not in existing_indices]
        if d_new:
            d_pca = reduced_pca[d_new]
            plt.scatter(d_pca[:, 0], d_pca[:, 1], 
                       s=100, marker='x', color='red', linewidth=3, 
                       zorder=8, label=f'D-optimal新規 ({len(d_new)})')

        i_new = [idx for idx in i_indices if idx not in existing_indices]
        if i_new:
            i_pca = reduced_pca[i_new]
            plt.scatter(i_pca[:, 0], i_pca[:, 1], 
                       s=100, marker='^', color='green', 
                       zorder=8, label=f'I-optimal新規 ({len(i_new)})')

        plt.title('PCA次元削減（UMAPフォールバック）', fontsize=14, weight='bold')
        plt.xlabel(f'第1主成分 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'第2主成分 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def main():
    """メイン実行関数（可視化機能強化版）"""
    print("🚀 化学実験計画法システム - 可視化機能強化版")
    print("="*60)
    print("📊 修正1: 各特徴量のヒストグラム色分け表示")
    print("📈 修正2: UMAP次元削減可視化（ハイパーパラメーター最適化機能付き）")
    print("="*60)

    # 設定読み込み
    try:
        design_df = pd.read_excel(SETTING_FILE, sheet_name=SHEET_DESIGN)
        info_df = pd.read_excel(SETTING_FILE, sheet_name=SHEET_INFO)
        n_experiments = int(info_df.loc[info_df["設定項目"] == "実験数", "値"].values[0])
        print(f"📋 設定読み込み完了")
        print(f"  - 説明変数数: {len(design_df)}")
        print(f"  - 目標実験数: {n_experiments}")
    except Exception as e:
        print(f"❌ 設定読み込みエラー: {e}")
        return

    # 候補点生成
    print(f"\n📊 候補点生成中...")
    candidate_points = generate_candidate_points(design_df)
    print(f"✅ 初期候補点生成完了: {len(candidate_points):,}点")

    # 既存実験データの処理
    existing_indices = []
    if USE_EXISTING_DATA:
        print(f"\n🔍 既存実験データ処理開始")

        existing_data, available_vars = load_and_validate_existing_data(
            EXISTING_DATA_FILE, design_df, verbose=True
        )

        if existing_data is not None and len(existing_data) > 0:
            variable_names = design_df["説明変数名"].tolist()
            existing_indices = match_existing_experiments_enhanced(
                candidate_points, existing_data, variable_names,
                tolerance_relative=1e-4,
                tolerance_absolute=1e-6,
                verbose=True
            )
        else:
            print("❌ 既存実験データが利用できません")

    # 候補点削減
    original_candidate_count = len(candidate_points)
    should_reduce = len(candidate_points) > CANDIDATE_REDUCTION_THRESHOLD

    if should_reduce:
        print(f"\n🔄 候補点削減実行: {original_candidate_count:,} → {MAX_REDUCED_CANDIDATES:,}")
        candidate_points, reduced_mapping = hierarchical_candidate_reduction(
            candidate_points, MAX_REDUCED_CANDIDATES, existing_indices
        )

        if existing_indices:
            existing_indices = [reduced_mapping.index(idx) for idx in existing_indices if idx in reduced_mapping]
            print(f"✅ 既存実験点マッピング更新: {len(existing_indices)}件保持")

    print(f"\n✅ 最終データセット:")
    print(f"  - 最終候補点数: {len(candidate_points):,}")
    print(f"  - 既存実験点数: {len(existing_indices)}")
    print(f"  - 既存実験活用率: {len(existing_indices)/n_experiments*100:.1f}%")

    # データ前処理
    candidate_df = pd.DataFrame(candidate_points, columns=design_df["説明変数名"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(candidate_points)
    print(f"✅ データ標準化完了")

    # D-optimal設計
    print(f"\n🎯 D-optimal設計実行")
    d_indices, d_score = select_d_optimal_design_enhanced(
        X_scaled, existing_indices, n_experiments, verbose=VERBOSE
    )
    print(f"✅ D-optimal設計完了")
    print(f"  - 最終スコア: {d_score:.4f}")
    print(f"  - 総選定点数: {len(d_indices)}")
    print(f"  - 既存点: {len([i for i in d_indices if i in existing_indices])}点")
    print(f"  - 新規点: {len([i for i in d_indices if i not in existing_indices])}点")

    # I-optimal設計
    print(f"\n🎯 I-optimal設計実行")
    i_indices = select_i_optimal_design(X_scaled, n_experiments, existing_indices)
    print(f"✅ I-optimal設計完了")
    print(f"  - 総選定点数: {len(i_indices)}")
    print(f"  - 既存点: {len([i for i in i_indices if i in existing_indices])}点")
    print(f"  - 新規点: {len([i for i in i_indices if i not in existing_indices])}点")

    # 結果処理（新規実験点のみ抽出）
    d_new_indices = [idx for idx in d_indices if idx not in existing_indices]
    i_new_indices = [idx for idx in i_indices if idx not in existing_indices]

    selected_d_df = candidate_df.iloc[d_new_indices] if d_new_indices else pd.DataFrame()
    selected_i_df = candidate_df.iloc[i_new_indices] if i_new_indices else pd.DataFrame()

    print(f"\n📊 選定結果サマリー:")
    print(f"  - 既存実験点活用: {len(existing_indices)}点")
    print(f"  - D-optimal新規選定: {len(d_new_indices)}点")
    print(f"  - I-optimal新規選定: {len(i_new_indices)}点")
    print(f"  - D-optimal総実験点: {len(d_indices)}点")
    print(f"  - I-optimal総実験点: {len(i_indices)}点")

    # ファイル保存
    if len(selected_d_df) > 0:
        selected_d_df.to_excel("D_optimal_新規実験点.xlsx", index=False)
    if len(selected_i_df) > 0:
        selected_i_df.to_excel("I_optimal_新規実験点.xlsx", index=False)

    all_d_df = candidate_df.iloc[d_indices].copy()
    all_i_df = candidate_df.iloc[i_indices].copy()
    all_d_df['データ種別'] = ['既存' if i in existing_indices else '新規' for i in d_indices]
    all_i_df['データ種別'] = ['既存' if i in existing_indices else '新規' for i in i_indices]
    all_d_df.to_excel("D_optimal_全実験点.xlsx", index=False)
    all_i_df.to_excel("I_optimal_全実験点.xlsx", index=False)

    candidate_df.to_excel("候補点一覧_v2.xlsx", index=False)
    print(f"💾 結果ファイル保存完了")

    # =============== 📊 可視化実行 ===============
    variable_names = design_df["説明変数名"].tolist()

    # 修正1: 特徴量ヒストグラム表示
    visualize_feature_histograms(candidate_df, d_indices, i_indices, existing_indices, variable_names)

    # 修正2: UMAP次元削減可視化
    visualize_umap_enhanced(X_scaled, d_indices, i_indices, existing_indices, variable_names)

    print(f"\n🎉 化学実験計画法システム完了（可視化機能強化版）")
    print("="*60)
    print("✅ 既存実験点を活用した最適実験計画が完成しました")
    print(f"📁 新規実験点ファイル: D_optimal_新規実験点.xlsx, I_optimal_新規実験点.xlsx")
    print(f"📁 全実験点ファイル: D_optimal_全実験点.xlsx, I_optimal_全実験点.xlsx")
    print("📊 可視化: 特徴量分布ヒストグラム + UMAP次元削減")
    print("="*60)

if __name__ == "__main__":
    main()



# In[ ]:




