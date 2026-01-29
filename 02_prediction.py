import os
import sys
from pathlib import Path

# プロジェクトルート設定
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pythonコードフォルダをパスに追加
PYTHON_CODE_FOLDER = PROJECT_ROOT / "00_Pythonコード"
if str(PYTHON_CODE_FOLDER) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_FOLDER))

import pandas as pd
import numpy as np
import joblib
from config import Config

# 日本語フォント設定
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = Config.JAPANESE_FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = Config.JAPANESE_FONT_UNICODE_MINUS

# --- 実行ファイル識別ログ（複製ディレクトリやカーネル混線の早期発見用） ---
import hashlib, time
def _print_self_identity():
    try:
        p = os.path.realpath(__file__)
        with open(p, 'rb') as f:
            h = hashlib.sha1(f.read()).hexdigest()[:10]
        mtime = time.ctime(os.path.getmtime(p))
        print(f"[RUNNING] {p}  mtime={mtime}  sha1={h}")
    except Exception as _e:
        # 失敗しても処理は継続
        pass
_print_self_identity()

# === Fail-Fast===
def assert_config_minimum(Config):
    required = [
        "RESULT_FOLDER","MODEL_FOLDER","PREDICTION_FOLDER","PREDICTION_DATA",
        "TARGET_COLUMNS","FEATURE_COLUMNS","RANDOM_STATE","get_n_jobs"
    ]
    missing = [k for k in required if not hasattr(Config, k)]
    if missing:
        raise RuntimeError(f"Config不足: {missing}")
    n_jobs = int(Config.get_n_jobs("default"))
    if n_jobs < 1:
        raise RuntimeError("Config.get_n_jobs('default') は 1以上が必要です。")

def assert_some_features_exist(df, feature_cols):
    exists = [c for c in feature_cols if c in df.columns]
    if not exists:
        raise RuntimeError("FEATURE_COLUMNS に該当する列が1つも予測データに存在しません。")
    return exists

# ===== ユーザ設定 =====
RESULT_FOLDER = Config.RESULT_FOLDER
TARGET_COLUMNS = Config.TARGET_COLUMNS
PREDICTION_FOLDER = Config.PREDICTION_FOLDER
PREDICTION_DATA = Config.PREDICTION_DATA
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

assert_config_minimum(Config)


# === robust loader for saved pipelines (A+E) ===
def _describe_saved_obj(obj):
    if hasattr(obj, "transform"):
        return "sklearn_pipeline"
    if isinstance(obj, dict):
        if "sk_pipeline" in obj:
            return "dict+sk_pipeline"
        return "dict"
    return type(obj).__name__

def apply_saved_pipeline(X_new: pd.DataFrame, bundle: dict, pipeline_path: str) -> np.ndarray:
    """
    dict / sklearn.Pipeline / bundle(preprocessor+selector) の順で適用。
    方式E: schema_version / pipeline_format があればログに出す。
    """
    X_tmp = X_new.copy()

    # 1) pipeline_{target}.pkl を試す
    if os.path.exists(pipeline_path):
        pipe_obj = joblib.load(pipeline_path)
        fmt = _describe_saved_obj(pipe_obj)
        schema = None
        if isinstance(pipe_obj, dict):
            schema = pipe_obj.get("schema_version", None)
        print(f"ℹ 既存パイプライン検出: format={fmt}" + (f", schema={schema}" if schema else ""))

        # 1-1) sklearn.Pipeline を最優先
        if hasattr(pipe_obj, "transform"):
            return pipe_obj.transform(X_tmp)

        # 1-2) dict（sk_pipeline が同梱されていればそれを優先）
        if isinstance(pipe_obj, dict):
            # a) sk_pipeline があればそれを使う
            if "sk_pipeline" in pipe_obj and hasattr(pipe_obj["sk_pipeline"], "transform"):
                return pipe_obj["sk_pipeline"].transform(X_tmp)

            # b) pre + sel が両方あれば順に適用
            pre = pipe_obj.get("preprocessor", None)
            sel = pipe_obj.get("selector", None)

            has_pre = (pre is not None and hasattr(pre, "transform"))
            has_sel = (sel is not None and hasattr(sel, "transform"))

            if has_pre and has_sel:
                X_tmp = pre.transform(X_tmp)
                X_tmp = sel.transform(X_tmp)
                return X_tmp

            if has_pre and not has_sel:
                # 前処理だけある → 前処理まで適用して返す（selector不要な設計も許容）
                X_tmp = pre.transform(X_tmp)
                return X_tmp

            if (not has_pre) and has_sel:
                # selector だけある → 危険（前処理前の空間に selector を当てる恐れ）
                # → ここでは使わず、下の bundle フォールバックに任せる
                print("ℹ dict内にselectorのみ検出 → bundleのpreprocessor/selectorへフォールバックします。")
                # 何もreturnしない（この後のbundle処理に進む）


    # 2) バンドル（final_model_*.pkl）へフォールバック
    print("ℹ pipelineファイル未使用 → bundle 内の preprocessor/selector を適用します。")
    pre = bundle.get("preprocessor", None)
    sel = bundle.get("selector", None)
    if pre is None or sel is None:
        raise RuntimeError("前処理オブジェクトが見つからず transform できません。")
    X_tmp = pre.transform(X_tmp) if hasattr(pre, "transform") else X_tmp
    X_tmp = sel.transform(X_tmp) if hasattr(sel, "transform") else X_tmp
    return X_tmp


# ===== 逆変換（学習時のmethod+shiftに厳密一致） =====
def inverse_transform_target(y_transformed: np.ndarray, method: str, shift: float = 0.0) -> np.ndarray:
    y_t = np.asarray(y_transformed, dtype=float)
    if method == 'log':
        y_orig = np.exp(y_t)
        return y_orig - float(shift) if shift else y_orig
    elif method == 'sqrt':
        y_orig = np.square(y_t)
        return y_orig - float(shift) if shift else y_orig
    else:
        return y_t    

# ===== 新データの読み込み =====
prediction_path = os.path.join(PREDICTION_FOLDER, PREDICTION_DATA)
if not os.path.exists(prediction_path):
    raise FileNotFoundError(f"予測データが見つかりません: {prediction_path}")

df_new = pd.read_excel(prediction_path)

print(f"✅ 予測データ読込: {df_new.shape} from {prediction_path}")

# ===== 切削時間の自動計算（列が無い場合のみ） =====
if Config.CUTTING_TIME_COLUMN_NAME not in df_new.columns:
    print(f"🔧 {Config.CUTTING_TIME_COLUMN_NAME}列を自動計算中...")
    if '送り速度' in df_new.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            calc = (Config.CUTTING_DISTANCE_MM / df_new['送り速度'] * 60.0)
        df_new[Config.CUTTING_TIME_COLUMN_NAME] = np.round(calc.astype(float), 1)
        print(f"✅ {Config.CUTTING_TIME_COLUMN_NAME}列を追加しました")
        try:
            _min = df_new[Config.CUTTING_TIME_COLUMN_NAME].min()
            _max = df_new[Config.CUTTING_TIME_COLUMN_NAME].max()
            print(f"   計算範囲: {(_min if np.isfinite(_min) else np.nan):.1f}～{(_max if np.isfinite(_max) else np.nan):.1f}秒")
        except Exception:
            pass
    else:
        print("⚠ 送り速度列が見つかりません。切削時間の計算をスキップします。")
else:
    print(f"ℹ {Config.CUTTING_TIME_COLUMN_NAME}列は既に存在します。計算をスキップします。")

# すべての予測結果を格納するDataFrame
df_out = df_new.copy()

# ===== ターゲットごとに予測 =====
for target in TARGET_COLUMNS:
    model_folder = Config.MODEL_FOLDER
    bundle_path = os.path.join(model_folder, f"{Config.FINAL_MODEL_PREFIX}_{target}.pkl")
    if not os.path.exists(bundle_path):
        print(f"⚠ バンドルが存在しません: {bundle_path}")
        continue

    print(f"\n=== {target} 用モデルで予測 ===")
    try:
        bundle = joblib.load(bundle_path)
    except Exception as e:
        print(f"❌ バンドル読込失敗: {e}")
        continue

    # 学習時保存物の取得
    model = bundle['final_model']
    feature_columns = bundle.get('feature_columns', Config.FEATURE_COLUMNS)
    transform_method = bundle.get('transform_method', 'none')
    transform_shift = float(bundle.get('transform_shift', 0.0))
    
    # ← ここでターゲットごとの特徴量存在をチェック
    _ = assert_some_features_exist(df_new, feature_columns)


    # 入力列を揃える（不足列は0で補完、余分は無視）＋ 数値化/型統一
    X_base = df_new.copy()
    for c in feature_columns:
        if c not in X_base.columns:
            X_base[c] = 0
    X_new = X_base[feature_columns].copy()
    # 型をfloat32に統一（前処理系と整合）
    for c in X_new.columns:
        # 数値化に失敗（例：文字列）しても0に落とす
        X_new[c] = pd.to_numeric(X_new[c], errors='coerce').fillna(0).astype(np.float32)

    # --- robust pipeline application（最小版・これだけでOK） ---
    pipeline_path = os.path.join(model_folder, f'pipeline_{target}.pkl')
    try:
        X_new_sel = apply_saved_pipeline(X_new, bundle, pipeline_path)

    # 行数が一致しない／0特徴量などの設定事故を fail-fast で検知
        shape_attr = getattr(X_new_sel, 'shape', None)
        if shape_attr is None or len(shape_attr) < 2:
            raise RuntimeError("前処理後のX_new_selにshape属性がありません。パイプライン保存形式を確認してください。")
        
        n_rows, n_cols = shape_attr[0], shape_attr[1]
        if n_rows != len(X_new):
            raise RuntimeError(f"前処理後の行数が一致しません: before={len(X_new)} after={n_rows}")
        
        if n_cols == 0:
            raise RuntimeError("前処理後の特徴量が0次元です。top_k や選択条件を見直してください。")
        
        print(f"ℹ 前処理後: X_new_sel shape={shape_attr}")
    
    except Exception as e:
        print(f"❌ 前処理に失敗: {e}")
        continue

    # 予測 → 逆変換（学習時method+shift）
    try:
        y_pred_trans = model.predict(X_new_sel)
    except Exception as e:
        print(f"❌ 予測失敗: {e}")
        continue

    y_pred = inverse_transform_target(y_pred_trans, transform_method, transform_shift)

    # 結果をDataFrameに追加
    df_out[f"{Config.PREDICTION_COLUMN_PREFIX}_{target}"] = y_pred

    # 予測統計を表示（任意）
    try:
        print(f"  予測件数: {len(y_pred)}")
        print(f"  予測範囲: [{np.nanmin(y_pred):.3f}, {np.nanmax(y_pred):.3f}]")
        print(f"  予測平均: {np.nanmean(y_pred):.3f}")
        print(f"  予測標準偏差: {np.nanstd(y_pred):.3f}")
    except Exception:
        pass

# ===== 保存 =====
out_path = os.path.join(PREDICTION_FOLDER, Config.PREDICTION_OUTPUT_FILE)
df_out.to_excel(out_path, index=False)
print(f"\n✅ 予測完了: {out_path}")
print(f"   出力列: {df_out.columns.tolist()}")

# __pycache__フォルダを99_Tempに移動
import shutil
import glob

def move_pycache_to_temp():
    temp_folder = Path("99_Temp")
    temp_folder.mkdir(exist_ok=True)
    pycache_folders = glob.glob("**/__pycache__", recursive=True)
    for pycache_path in pycache_folders:
        pycache_path = Path(pycache_path)
        if pycache_path.exists() and pycache_path.is_dir():
            try:
                relative_path = pycache_path.relative_to(Path.cwd())
                temp_dest = temp_folder / f"{relative_path.parent.name}__pycache__"
            except ValueError:
                temp_dest = temp_folder / f"{pycache_path.parent.name}__pycache__"
            if temp_dest.exists():
                shutil.rmtree(temp_dest)
            shutil.move(str(pycache_path), str(temp_dest))
            print(f"✅ {pycache_path} → {temp_dest}")

move_pycache_to_temp()
