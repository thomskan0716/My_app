import pandas as pd, pathlib

path = pathlib.Path(r'C:\Users\xebec0176\Desktop\0.00sec\.venv\Archivos_pruebas\●20251202_Nonlinear_regression_ver4 - コピー\01_データセット\filtered_data.xlsx')
df = pd.read_excel(path)
# Convierte todas las int64 a int32 (o a float si prefieres)
int_cols = df.dtypes[df.dtypes == 'int64'].index
df[int_cols] = df[int_cols].astype('int32')
# o: df[int_cols] = df[int_cols].astype('float32')
out = path.with_name('filtered_data_fixed.xlsx')
df.to_excel(out, index=False)
print("Guardado:", out)