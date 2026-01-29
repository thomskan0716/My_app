"""
推論エントリ（Jupyter/CLI両対応・安全版）：
- Jupyter が注入する '-f <connection.json>' を無視
- -i/--input, -o/--output を受け付け（任意）
- 位置引数にExcelがあれば採用、無ければ ConfigCLS.PREDICT_INPUT_FILE を使用
"""
import os
import sys
import argparse
from typing import Optional, List

from config_cls import ConfigCLS
from predict_cls import predict_excel


def _is_excel_path(p: str) -> bool:
    if not isinstance(p, str):
        return False
    low = p.lower()
    return low.endswith(".xlsx") or low.endswith(".xls")


def _pick_positional_excel(paths: List[str]) -> Optional[str]:
    """位置引数からExcelっぽいパスだけを拾う（存在チェックは後段で）"""
    for p in paths:
        if p and not p.startswith("-") and _is_excel_path(p):
            return p
    return None


def main():
    # ---- 引数パース（未知の引数は無視） ----
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", dest="input_path", default=None)
    parser.add_argument("-o", "--output", dest="output_path", default=None)
    parser.add_argument("positional", nargs="*")
    args, _unknown = parser.parse_known_args()

    # 1) 明示指定（-i/--input）を最優先
    input_path = args.input_path

    # 2) 位置引数にExcelがあれば採用（Jupyterの -f は無視される）
    if input_path is None:
        cand = _pick_positional_excel(args.positional)
        if cand:
            input_path = cand

    # 3) 最後に Config の既定を使う
    if input_path is None:
        input_path = os.path.join(ConfigCLS.DATA_FOLDER, ConfigCLS.PREDICT_INPUT_FILE)

    if input_path is None:
        raise ValueError("予測用Excelのパスを指定してください（-i/--input または ConfigCLS.PREDICT_INPUT_FILE）。")

    # 相対パスはカレントからの解決
    input_path = os.path.abspath(input_path)

    # Excel拡張子バリデーション
    if not _is_excel_path(input_path):
        raise ValueError(f"Excelファイルではありません: {input_path}")

    # 実在チェック（存在しなければ丁寧に案内）
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"予測用Excelが見つかりません: {input_path}\n"
            "Jupyter の自動引数（-f <connection.json>）を拾っている可能性があります。\n"
            "以下のいずれかで実行してください：\n"
            "  1) from main_predict_cls import main as predict_main; predict_main()\n"
            "  2) !python main_predict_cls.py -i your_input.xlsx -o out.xlsx\n"
            "  3) ConfigCLS.PREDICT_INPUT_FILE を既存ファイルに設定してから main() を呼ぶ\n"
        )

    # 出力パス：なければデフォルト（入力ファイル名に _pred を付与）
    output_path = args.output_path
    if output_path is None:
        base, ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(ConfigCLS.PREDICTION_FOLDER_PATH, f"{base}_pred{ext}")

    # 予測実行
    os.makedirs(ConfigCLS.PREDICTION_FOLDER_PATH, exist_ok=True)
    out = predict_excel(input_excel=input_path, output_excel=output_path)
    print(f"✅ 予測Excelを保存: {out}")


if __name__ == "__main__":
    main()
