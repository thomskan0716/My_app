#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os

def check_filtered_data():
    """Verificar el contenido del archivo filtered_data.xlsx"""
    try:
        # Ruta del archivo que menciona el usuario
        file_path = r"C:\Users\xebec0176\Desktop\0.00sec\.venv\Archivos_de_salida\Proyecto_78\03_線形回帰\27_20250903_193254\01_学習モデル\filtered_data.xlsx"
        
        print(f"🔧 Verificando archivo: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ Archivo no encontrado: {file_path}")
            return
        
        # Leer el archivo Excel
        df = pd.read_excel(file_path)
        print(f"📊 Filas en el archivo: {len(df)}")
        print(f"📊 Columnas en el archivo: {list(df.columns)}")
        
        # Verificar valores únicos en 線材長
        if '線材長' in df.columns:
            unique_values = df['線材長'].unique()
            print(f"📊 Valores únicos en 線材長: {unique_values}")
            
            # Mostrar distribución
            value_counts = df['線材長'].value_counts().sort_index()
            print(f"📊 Distribución de valores en 線材長:")
            for value, count in value_counts.items():
                print(f"   {value}: {count} registros")
            
            # Verificar si hay valores de 74
            if 74 in unique_values:
                print(f"❌ PROBLEMA CONFIRMADO: Se encontraron {value_counts[74]} registros con 線材長 = 74")
                
                # Mostrar algunos ejemplos de registros con 線材長 = 74
                df_74 = df[df['線材長'] == 74]
                print(f"\n📋 Ejemplos de registros con 線材長 = 74:")
                print(df_74.head(3).to_string())
                
                # Verificar si estos registros tienen A11 = 1
                if 'A11' in df.columns:
                    a11_counts = df_74['A11'].value_counts()
                    print(f"\n📊 Distribución de A11 en registros con 線材長 = 74:")
                    for value, count in a11_counts.items():
                        print(f"   A11 = {value}: {count} registros")
            else:
                print("✅ CORRECTO: No se encontraron registros con 線材長 = 74")
        else:
            print("❌ Columna 線材長 no encontrada en el archivo")
        
        # Verificar otras columnas relevantes
        relevant_cols = ['A11', 'A13', 'A21', 'A32']
        for col in relevant_cols:
            if col in df.columns:
                unique_vals = df[col].unique()
                print(f"📊 Valores únicos en {col}: {unique_vals}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_filtered_data()

