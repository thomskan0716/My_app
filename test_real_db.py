#!/usr/bin/env python
# coding: utf-8

"""
ES: Script para probar la función de análisis lineal con la base de datos real.
EN: Script to test the linear analysis function using the real database.
JA: 実DBで線形解析関数をテストするスクリプト。
"""

import os
import sys
from db_manager import DBManager
from linear_analysis_advanced import run_advanced_linear_analysis_from_db

def test_real_database():
    """ES: Probar la función con la base de datos real
    EN: Test the function with the real database
    JA: 実DBで関数をテスト
    """
    
    print("🔧 実DBで線形解析をテスト中...")
    
    try:
        # ES: Crear instancia de DBManager | EN: Create DBManager instance | JA: DBManager を作成
        db = DBManager()
        print("✅ DBManager を作成しました")
        
        # ES: Verificar tablas y datos | EN: Check tables and data | JA: テーブルとデータを確認
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📊 利用可能なテーブル: {[t[0] for t in tables]}")
        
        # ES: Verificar datos en cada tabla | EN: Check data in each table | JA: 各テーブルのデータ件数を確認
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"📈 {table_name}: {count} 件")
        
        # ES: Crear filtros de prueba | EN: Create test filters | JA: テスト用フィルタを作成
        test_filters = {
            'A32': True,  # Only brush A32
            '送り速度': ('20', '40'),  # Feed speed range
            '切込量': ('0.8', '1.5')  # Cut depth range
        }
        
        print(f"🔧 テストフィルタ: {test_filters}")
        
        # ES: Crear carpeta de salida | EN: Create output folder | JA: 出力フォルダを作成
        output_folder = "test_real_analysis"
        os.makedirs(output_folder, exist_ok=True)
        
        # ES: Ejecutar análisis | EN: Run analysis | JA: 解析を実行
        print("🚀 線形解析を実行中...")
        results = run_advanced_linear_analysis_from_db(db, test_filters, output_folder)
        
        # ES: Mostrar resultados | EN: Show results | JA: 結果表示
        print(f"✅ 結果: {results.get('success', False)}")
        
        if results.get('success', False):
            print(f"📊 処理データ数: {results.get('data_count', 0)}")
            print(f"🤖 Modelos entrenados: {results.get('models_trained', 0)}")
            print(f"📁 出力フォルダー: {results.get('output_folder', 'N/A')}")
            
            # ES: Mostrar resumen de modelos | EN: Show model summary | JA: モデル要約を表示
            summary = results.get('summary', [])
            if summary:
                print("📋 モデル要約:")
                for item in summary:
                    print(f"  - {item}")
        else:
            error_msg = results.get('error', '不明なエラー')
            print(f"❌ エラー: {error_msg}")
        
        print("✅ テスト完了")
        
    except Exception as e:
        print(f"❌ テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_database()
