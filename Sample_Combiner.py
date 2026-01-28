import pandas as pd
import numpy as np
from itertools import product
import os
import time


class SampleCombiner:
    """ES: Clase para generar todas las combinaciones posibles a partir de un archivo de parámetros.
    EN: Class to generate all possible combinations from a parameter file.
    JA: パラメータファイルから全組合せを生成するクラス。
    """

    @staticmethod
    def generate_combinations(input_path: str, output_path: str):
        """
        ES: Lee el archivo de configuración inicial, genera combinaciones y exporta SIEMPRE:
        EN: Read the initial config file, generate combinations, and ALWAYS export:
        JA: 初期設定ファイルを読み込み、組合せを生成して常に出力：
        ES: - CSV (utf-8-sig)
        EN: - CSV (utf-8-sig)
        JA: - CSV（utf-8-sig）
        ES: - Excel (si >500,000 filas, se divide en varios excels de 500,000 filas dentro de una carpeta)
        EN: - Excel (if >500,000 rows, split into multiple 500,000-row Excel files inside a folder)
        JA: - Excel（50万行超なら、フォルダ内に50万行ごとに分割して複数Excel出力）
        ES: Además muestra logs de progreso/ETA.
        EN: Also prints progress/ETA logs.
        JA: 進捗/残り時間（ETA）ログも表示。
        """

        df = pd.read_excel(input_path, index_col=0)  # Use first column as index

        params = {}
        brush_values = None

        for column in df.columns:
            min_val = df.at['min', column]
            max_val = df.at['max', column]
            interval = df.at['間隔', column]

            # ES: Normalizar nombre de columna
            # EN: Normalize column name
            # JA: 列名を正規化
            normalized_column = "突出量" if column == "突出し量" else column

            # ES: Brush: especial -> producir A13/A11/A21/A32 (one-hot)
            # EN: Brush: special -> produce A13/A11/A21/A32 (one-hot)
            # JA: ブラシ：特別扱い -> A13/A11/A21/A32（one-hot）を生成
            if normalized_column == "ブラシ":
                try:
                    if isinstance(interval, str) and "," in interval:
                        brush_values = [int(float(x.strip())) for x in interval.split(",")]
                    else:
                        bmin = int(float(min_val))
                        bmax = int(float(max_val))
                        brush_values = list(range(bmin, bmax + 1))
                except Exception as e:
                    raise ValueError(f"⚠️ Error procesando 'ブラシ': min={min_val}, max={max_val}, interval={interval}\n{e}")
                continue

            try:
                # ES: Caso categórico detectado por coma o texto "なし"
                # EN: Categorical case detected via comma-separated values or "なし"
                # JA: カテゴリ値（カンマ区切り、または「なし」）を検出
                if isinstance(interval, str) and ("," in interval or interval.strip() == "なし"):
                    values = [float(x.strip()) for x in interval.split(",")]
                else:
                    min_val = float(min_val)
                    max_val = float(max_val)
                    interval = float(interval)
                    values = np.arange(min_val, max_val + interval, interval)
            except Exception as e:
                raise ValueError(
                    f"⚠️ Error procesando '{column}': min={min_val}, max={max_val}, interval={interval}\n{e}")

            params[normalized_column] = values

        if brush_values is None:
            # ES: Si no se especifica ブラシ, asumir todos (por compatibilidad)
            # EN: If ブラシ is not specified, assume all (for compatibility)
            # JA: ブラシが未指定なら全種類を仮定（互換性のため）
            brush_values = [1, 2, 3, 4]

        # ES: Determinar paths de salida: base + (.csv y .xlsx)
        # EN: Determine output paths: base + (.csv and .xlsx)
        # JA: 出力パスを決定：base +（.csv と .xlsx）
        base, ext = os.path.splitext(output_path)
        if ext.lower() in (".xlsx", ".xls", ".csv"):
            base_path = base
        else:
            base_path = output_path
        csv_path = base_path + ".csv"
        excel_path = base_path + ".xlsx"

        # ES: Conteo total para logs/particionado
        # EN: Total row count for logs/splitting
        # JA: ログ/分割用の総行数を算出
        total_rows = 1
        for v in params.values():
            total_rows *= len(v)
        total_rows *= len(brush_values)

        rows_per_excel = 500_000
        chunksize = 100_000

        print(f"✅ 生成する組み合わせ: {total_rows:,} 行", flush=True)
        print(f"📄 CSV 出力: {csv_path}", flush=True)
        if total_rows <= rows_per_excel:
            print(f"📄 Excel 出力: {excel_path}", flush=True)
        else:
            excel_folder = base_path + "_excel_parts"
            print(f"📁 Excel 分割: {excel_folder}（500,000 行/ファイル）", flush=True)

        # Preparar columnas (brush one-hot primero)
        other_cols = list(params.keys())
        out_cols = ["A13", "A11", "A21", "A32"] + other_cols

        # Writers
        wrote_header_csv = False
        start_ts = time.time()
        written = 0

        # Excel writing state
        excel_folder = None
        writer = None
        startrow = 0
        part_rows = 0
        part_idx = 1

        def _open_excel_writer():
            nonlocal writer, startrow, part_rows, part_idx, excel_folder
            if writer is not None:
                writer.close()
            if excel_folder is None:
                excel_folder = base_path + "_excel_parts"
                os.makedirs(excel_folder, exist_ok=True)
            part_path = os.path.join(excel_folder, f"{os.path.basename(base_path)}_part_{part_idx:03d}.xlsx")
            part_idx += 1
            startrow = 0
            part_rows = 0
            writer = pd.ExcelWriter(part_path, engine="openpyxl")
            print(f"📄 Escribiendo {os.path.basename(part_path)}", flush=True)

        def _write_excel_chunk(chunk_df: pd.DataFrame):
            nonlocal writer, startrow, part_rows
            if total_rows <= rows_per_excel:
                # Un solo excel
                # ES: Abrir si no existe
                # EN: Open it if it doesn't exist
                # JP: 存在しなければ開く
                if writer is None:
                    writer = pd.ExcelWriter(excel_path, engine="openpyxl")
                    startrow = 0
                    part_rows = 0
                header = startrow == 0
                chunk_df.to_excel(writer, index=False, header=header, startrow=startrow, sheet_name="Sheet1")
                startrow += len(chunk_df) + (1 if header else 0)
                part_rows += len(chunk_df)
            else:
                # Multiparts
                if writer is None or part_rows >= rows_per_excel:
                    _open_excel_writer()
                pos = 0
                while pos < len(chunk_df):
                    if part_rows >= rows_per_excel:
                        _open_excel_writer()
                    remaining = rows_per_excel - part_rows
                    take = min(remaining, len(chunk_df) - pos)
                    piece = chunk_df.iloc[pos:pos + take]
                    header = startrow == 0
                    piece.to_excel(writer, index=False, header=header, startrow=startrow, sheet_name="Sheet1")
                    startrow += take + (1 if header else 0)
                    part_rows += take
                    pos += take

        # ES: Si el CSV ya existe, borrarlo para no acumular filas de ejecuciones anteriores (mode="a")
        # EN: If the CSV already exists, remove it so we don't accumulate rows from previous runs (mode="a")
        # JP: 既存CSVがあれば削除し、前回実行分が追記されないようにする（mode="a"のため）
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
            except OSError as e:
                raise IOError(f"⚠️ No se pudo borrar el CSV previo para escritura limpia: {csv_path}\n{e}") from e

        # ES: Generación en streaming
        # EN: Streaming generation
        # JA: ストリーミング生成
        chunk_rows = []

        # Map brush value -> one-hot
        # 1->A11, 2->A21, 3->A32, 4->A13
        def _brush_one_hot(b: int):
            return {
                "A13": 1 if b == 4 else 0,
                "A11": 1 if b == 1 else 0,
                "A21": 1 if b == 2 else 0,
                "A32": 1 if b == 3 else 0,
            }

        values_lists = list(params.values())
        for b in brush_values:
            onehot = _brush_one_hot(int(b))
            for combo in product(*values_lists):
                row = [onehot["A13"], onehot["A11"], onehot["A21"], onehot["A32"], *combo]
                chunk_rows.append(row)
                if len(chunk_rows) >= chunksize:
                    chunk_df = pd.DataFrame(chunk_rows, columns=out_cols)
                    # CSV append
                    chunk_df.to_csv(csv_path, mode="a", index=False, header=not wrote_header_csv, encoding="utf-8-sig")
                    wrote_header_csv = True
                    # Excel write
                    _write_excel_chunk(chunk_df)

                    written += len(chunk_df)
                    chunk_rows = []

                    # Logs
                    elapsed = max(time.time() - start_ts, 1e-9)
                    speed = written / elapsed
                    remaining = max(total_rows - written, 0)
                    eta_s = remaining / speed if speed > 0 else float("inf")
                    pct = (written / total_rows * 100.0) if total_rows else 100.0
                    print(f"⏳ {pct:6.2f}%  {written:,}/{total_rows:,}  {speed:,.0f} rows/s  ETA {eta_s/60:.1f} min", flush=True)

        # flush final
        if chunk_rows:
            chunk_df = pd.DataFrame(chunk_rows, columns=out_cols)
            chunk_df.to_csv(csv_path, mode="a", index=False, header=not wrote_header_csv, encoding="utf-8-sig")
            _write_excel_chunk(chunk_df)
            written += len(chunk_df)

        if writer is not None:
            writer.close()

        print(f"✅ 組み合わせを生成してエクスポートしました: {written:,} 行", flush=True)
