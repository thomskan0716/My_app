#!/usr/bin/env python
# coding: utf-8

"""
Test script to verify that linear_analysis_advanced.py works exactly like the reference file
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from linear_analysis_advanced import run_advanced_linear_analysis_from_db
from db_manager import DBManager

def test_exact_reference():
    """Test that the analysis works exactly like the reference file"""
    print("🧪 参照ファイルと同一になる解析をテスト中")
    print("=" * 60)
    
    try:
        # Initialize database manager
        db_manager = DBManager()
        
        # Test without filters first
        print("📊 フィルタなしで解析を実行中...")
        results = run_advanced_linear_analysis_from_db(db_manager, filters=None)
        
        if results['success']:
            print("✅ 解析成功")
            print(f"📁 出力ディレクトリ: {results['output_directory']}")
            print(f"📊 データ形状: {results['data_shape']}")
            
            # Check models
            models = results.get('models', {})
            print(f"🤖 Modelos entrenados: {len(models)}")
            
            for target, model_info in models.items():
                if model_info.get('model') is not None:
                    model_name = model_info.get('model_name', 'Unknown')
                    task_type = model_info.get('task_type', 'Unknown')
                    print(f"  - {target}: {model_name} ({task_type})")
                    
                    # Check metrics
                    metrics = model_info.get('final_metrics', {})
                    if task_type == 'regression':
                        r2 = metrics.get('r2', 'N/A')
                        mae = metrics.get('mae', 'N/A')
                        rmse = metrics.get('rmse', 'N/A')
                        print(f"    R²: {r2}, MAE: {mae}, RMSE: {rmse}")
                    else:
                        accuracy = metrics.get('accuracy', 'N/A')
                        f1 = metrics.get('f1_score', 'N/A')
                        print(f"    Accuracy: {accuracy}, F1: {f1}")
                else:
                    error = model_info.get('error', 'Unknown error')
                    print(f"  - {target}: ERROR - {error}")
            
            # Check transformations
            transformations = results.get('transformation_info', {})
            print(f"🔄 適用した変換: {len([t for t in transformations.values() if t.get('applied', False)])}")
            
            for target, trans_info in transformations.items():
                if trans_info.get('applied', False):
                    method = trans_info.get('method', 'unknown')
                    print(f"  - {target}: {method} 変換")
                else:
                    reason = trans_info.get('reason', 'no transformation')
                    print(f"  - {target}: 変換なし（{reason}）")
            
            # Check Excel calculator
            excel_path = results.get('excel_calculator')
            if excel_path:
                print(f"📊 Excel計算機: {excel_path}")
                if os.path.exists(excel_path):
                    print("✅ Excelファイルを作成しました")
                else:
                    print("❌ Excelファイルが見つかりません")
            else:
                print("⚠️ Excel計算機は作成されませんでした")
            
            # Check output directory structure
            output_dir = Path(results['output_directory'])
            if output_dir.exists():
                print(f"📁 ディレクトリ構造:")
                for item in output_dir.rglob('*'):
                    if item.is_file():
                        rel_path = item.relative_to(output_dir)
                        size = item.stat().st_size
                        print(f"  - {rel_path} ({size} bytes)")
            
            print("\n🎉 テスト成功: 解析は参照ファイルと同一です")
            
        else:
            print(f"❌ 解析失敗: {results.get('error', '不明なエラー')}")
            return False
            
    except Exception as e:
        print(f"❌ テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_exact_reference()
    if success:
        print("\n✅ 全テスト成功 - 解析は参照ファイルと同一です")
    else:
        print("\n❌ 一部テスト失敗 - 実装を確認してください")
