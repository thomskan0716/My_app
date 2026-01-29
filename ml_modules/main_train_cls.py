"""
学習エントリ：
- Excelを読み込み
- DCV＋校正＋しきい値最適化＋OOD学習
- バンドル保存
"""
import os
from config_cls import ConfigCLS
from trainer_dcv_cls import train_and_bundle


def main():
    ConfigCLS.validate()
    os.makedirs(ConfigCLS.MODEL_FOLDER_PATH, exist_ok=True)
    bundle = train_and_bundle(os.path.join(ConfigCLS.DATA_FOLDER, ConfigCLS.INPUT_FILE))
    print("=== 学習完了 ===")
    print(f"tau_pos = {bundle['tau_pos']:.6f} / tau_neg = {bundle['tau_neg']:.6f}")
    print(f"calibrator = {bundle['calibrator_name']}")
    print(f"selected_features = {len(bundle['selected_columns'])} columns")
    print(f"bundle saved to: {os.path.join(ConfigCLS.MODEL_FOLDER_PATH, 'final_bundle_cls.pkl')}")

if __name__ == "__main__":
    main()