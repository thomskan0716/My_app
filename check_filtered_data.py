#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os

def check_filtered_data():
    """ES: Verificar el contenido del archivo filtered_data.xlsx
    EN: Verify the contents of the filtered_data.xlsx file
    JP: filtered_data.xlsx の内容を確認する
    """
    try:
        # ES: Ruta del archivo que menciona el usuario
        # EN: Path to the file mentioned by the user
        # JP: ユーザーが言及したファイルのパス
        file_path = r"C:\Users\xebec0176\Desktop\0.00sec\.venv\Archivos_de_salida\Proyecto_78\03_線形回帰\27_20250903_193254\01_学習モデル\filtered_data.xlsx"
        
        print(f"🔧 ファイルを確認中: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ ファイルが見つかりません: {file_path}")
            return
        
        # ES: Leer el archivo Excel
        # EN: Read the Excel file
        # JP: Excelファイルを読み込む
        df = pd.read_excel(file_path)
        print(f"📊 行数: {len(df)}")
        print(f"📊 列: {list(df.columns)}")
        
        # Verificar valores únicos en 線材長
        if '線材長' in df.columns:
            unique_values = df['線材長'].unique()
            print(f"📊 線材長 のユニーク値: {unique_values}")
            
            # Mostrar distribución
            value_counts = df['線材長'].value_counts().sort_index()
            print(f"📊 線材長 の分布:")
            for value, count in value_counts.items():
                print(f"   {value}: {count} 件")
            
            # ES: Verificar si hay valores de 74
            # EN: Check whether there are values equal to 74
            # JP: 74の値があるか確認する
            if 74 in unique_values:
                print(f"❌ 問題を確認: 線材長 = 74 のレコードが {value_counts[74]} 件見つかりました")
                
                # Mostrar algunos ejemplos de registros con 線材長 = 74
                df_74 = df[df['線材長'] == 74]
                print(f"\n📋 線材長 = 74 のレコード例:")
                print(df_74.head(3).to_string())
                
                # ES: Verificar si estos registros tienen A11 = 1
                # EN: Check whether these rows have A11 = 1
                # JP: これらの行でA11 = 1かどうか確認する
                if 'A11' in df.columns:
                    a11_counts = df_74['A11'].value_counts()
                    print(f"\n📊 線材長 = 74 のレコードにおける A11 の分布:")
                    for value, count in a11_counts.items():
                        print(f"   A11 = {value}: {count} 件")
            else:
                print("✅ OK: 線材長 = 74 のレコードは見つかりませんでした")
        else:
            print("❌ ファイルに列 線材長 がありません")
        
        # ES: Verificar otras columnas relevantes
        # EN: Check other relevant columns
        # JP: 他の関連列を確認する
        relevant_cols = ['A11', 'A13', 'A21', 'A32']
        for col in relevant_cols:
            if col in df.columns:
                unique_vals = df[col].unique()
                print(f"📊 {col} のユニーク値: {unique_vals}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_filtered_data()

