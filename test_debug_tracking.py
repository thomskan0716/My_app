#!/usr/bin/env python
# coding: utf-8

from db_manager import DBManager
from linear_analysis_advanced import run_advanced_linear_analysis_from_db
import pandas as pd
import os

def test_debug_tracking():
    """ES: Testear el tracking de datos con debugs
    EN: Test data tracking with debug output
    JA: デバッグ出力付きでデータ追跡をテスト
    """
    try:
        print("🔧 デバッグ出力付きでデータ追跡をテスト中...")
        print("=" * 70)
        
        # ES: Crear instancia de DBManager | EN: Create DBManager instance | JA: DBManager を作成
        db_manager = DBManager()
        
        # ES: Filtros que deberían devolver solo registros con 線材長 = 75
        # EN: Filters that should return only records with 線材長 = 75
        # JA: 線材長=75 のレコードのみ返すはずのフィルタ
        filters = {
            'A11': True,
            '線材長': ('75', '75')
        }
        
        print(f"📊 適用フィルタ: {filters}")
        print("=" * 70)
        
        # ES: Ejecutar análisis lineal con debugs
        # EN: Run linear analysis with debug output
        # JA: デバッグ出力付きで線形解析を実行
        results = run_advanced_linear_analysis_from_db(
            db_manager, 
            filters, 
            "debug_tracking_test"
        )
        
        print("=" * 70)
        print("📋 最終検証:")
        print("=" * 70)
        
        if results.get('success', False):
            print(f"✅ 解析成功")
            print(f"📊 処理データ数: {results.get('data_count', 0)}")
            print(f"📊 データ範囲: {results.get('data_range', 'N/A')}")
            
            # ES: Verificar el archivo Excel final | EN: Verify final Excel file | JA: 最終Excelファイルを確認
            excel_path = os.path.join("debug_tracking_test", "01_学習モデル", "filtered_data.xlsx")
            if os.path.exists(excel_path):
                df_final = pd.read_excel(excel_path)
                print(f"\n📊 最終Excelファイルの検証:")
                print(f"📊 最終ファイルの行数: {len(df_final)}")
                
                if '線材長' in df_final.columns:
                    unique_final = df_final['線材長'].unique()
                    print(f"📊 線材長 の最終ユニーク値: {unique_final}")
                    
                    count_74_final = len(df_final[df_final['線材長'] == 74])
                    count_75_final = len(df_final[df_final['線材長'] == 75])
                    print(f"📊 線材長 = 74 の最終レコード数: {count_74_final}")
                    print(f"📊 線材長 = 75 の最終レコード数: {count_75_final}")
                    
                    if count_74_final > 0:
                        print(f"❌ 問題: 最終ファイルに 線材長 = 74 のレコードが {count_74_final} 件含まれています")
                    else:
                        print(f"✅ OK: 最終ファイルは 線材長 = 75 のレコードのみです")
                else:
                    print("❌ 最終ファイルに列 線材長 がありません")
            else:
                print(f"❌ 最終Excelファイルが見つかりません: {excel_path}")
        else:
            print(f"❌ 解析エラー: {results.get('error', '不明なエラー')}")
        
        db_manager.conn.close()
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug_tracking()

