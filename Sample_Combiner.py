import pandas as pd
import numpy as np
from itertools import product
import os
import time


class SampleCombiner:
    """Clase para generar todas las combinaciones posibles a partir de un archivo de parámetros."""

    @staticmethod
    def generate_combinations(input_path: str, output_path: str):
        """
        Lee el archivo de configuración inicial, genera combinaciones, y exporta SIEMPRE:
        - CSV (utf-8-sig)
        - Excel (si >500,000 filas, se divide en varios excels de 500,000 filas dentro de una carpeta)
        Además muestra logs de progreso/ETA.
        """

        df = pd.read_excel(input_path, index_col=0)  # Usar primera columna como índice

        params = {}
        brush_values = None

        for column in df.columns:
            min_val = df.at['min', column]
            max_val = df.at['max', column]
            interval = df.at['間隔', column]

            # Normalizar nombre de columna
            normalized_column = "突出量" if column == "突出し量" else column

            # Brush: especial -> producir A13/A11/A21/A32 (one-hot)
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
                # Caso categórico detectado por coma o texto "なし"
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
            # Si no se especifica 브라시, asumir todos (por compatibilidad)
            brush_values = [1, 2, 3, 4]

        # Determinar paths de salida: base + (.csv y .xlsx)
        base, ext = os.path.splitext(output_path)
        if ext.lower() in (".xlsx", ".xls", ".csv"):
            base_path = base
        else:
            base_path = output_path
        csv_path = base_path + ".csv"
        excel_path = base_path + ".xlsx"

        # Conteo total para logs/particionado
        total_rows = 1
        for v in params.values():
            total_rows *= len(v)
        total_rows *= len(brush_values)

        rows_per_excel = 500_000
        chunksize = 100_000

        print(f"✅ Combinaciones a generar: {total_rows:,} filas", flush=True)
        print(f"📄 CSV salida: {csv_path}", flush=True)
        if total_rows <= rows_per_excel:
            print(f"📄 Excel salida: {excel_path}", flush=True)
        else:
            excel_folder = base_path + "_excel_parts"
            print(f"📁 Excel parts: {excel_folder} (500,000 filas/archivo)", flush=True)

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
                # Abrir si no existe
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

        # Generación en streaming
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

        print(f"✅ Combinaciones generadas y exportadas: {written:,} filas", flush=True)
