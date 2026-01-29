#!/usr/bin/env python
# coding: utf-8

from db_manager import DBManager
from linear_analysis_advanced import run_advanced_linear_analysis_from_db

def test_filter_logic():
    """ES: Probar la lógica de filtros corregida
    EN: Test the corrected filter logic
    JA: 修正済みフィルタロジックをテスト
    """
    try:
        print("🔧 修正済みフィルタロジックをテスト中...")
        
        db = DBManager()
        
        # ES: Crear carpeta de prueba | EN: Create test folder | JA: テスト用フォルダを作成
        import os
        test_folder = "test_filter_logic"
        os.makedirs(test_folder, exist_ok=True)
        
        # ES: Casos de prueba específicos | EN: Specific test cases | JA: 具体的なテストケース
        test_cases = [
            {
                'name': '線材長 の範囲のみ (75-75)',
                'filters': {
                    '線材長': ('75', '75')
                },
                'expected': '線材長 = 75 のレコード'
            },
            {
                'name': 'A11 のみ',
                'filters': {
                    'A11': True
                },
                'expected': 'A11 = 1 のレコード'
            },
            {
                'name': 'すべて を選択',
                'filters': {
                    'すべて': True
                },
                'expected': 'いずれかのブラシ列 = 1 のレコード'
            },
            {
                'name': 'ブラシ指定なしの速度範囲',
                'filters': {
                    '送り速度': ('1000', '2000')
                },
                'expected': '送り速度 1000-2000 のレコード'
            },
            {
                'name': 'A11 + 速度範囲',
                'filters': {
                    'A11': True,
                    '送り速度': ('1000', '2000')
                },
                'expected': 'A11=1 かつ 送り速度 1000-2000 のレコード'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*60}")
            print(f"🔧 テスト {i+1}: {test_case['name']}")
            print(f"フィルタ: {test_case['filters']}")
            print(f"期待結果: {test_case['expected']}")
            print(f"{'='*60}")
            
            # ES: Ejecutar análisis | EN: Run analysis | JA: 解析を実行
            results = run_advanced_linear_analysis_from_db(
                db, 
                test_case['filters'], 
                test_folder
            )
            
            if results.get('success', False):
                data_count = results.get('data_count', 0)
                models_trained = results.get('models_trained', 0)
                print(f"✅ 成功: 処理データ {data_count} 件, 学習モデル {models_trained} 件")
                
                # Mostrar resumen de modelos
                summary = results.get('summary', [])
                if summary:
                    print("📊 学習したモデル:")
                    for item in summary:
                        target = item['target']
                        model = item['model']
                        if 'r2' in item:
                            metric = f"R² = {item['r2']:.3f}"
                        else:
                            metric = f"Accuracy = {item['accuracy']:.3f}"
                        print(f"   {target}: {model} - {metric}")
                
            else:
                error = results.get('error', '不明なエラー')
                print(f"❌ エラー: {error}")
        
        print(f"\n{'='*60}")
        print("✅ 全テスト完了")
        print(f"📁 結果の保存先: {test_folder}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_filter_logic()
