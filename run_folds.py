"""
ES: Launcher para ejecutar cada fold en un proceso separado.
EN: Launcher to run each fold in a separate process.
JA: 各foldを別プロセスで実行するランチャー。

ES: Esto evita acumulación de fragmentación del heap nativo entre folds.
EN: This avoids native-heap fragmentation accumulating across folds.
JA: fold間でネイティブヒープ断片化が蓄積するのを防ぐ。
"""
import os
import sys
import subprocess
from pathlib import Path

# ES: Obtener ruta del script desde el directorio actual
# EN: Resolve script path from current directory
# JA: 現在ディレクトリからスクリプトパスを解決
SCRIPT_DIR = Path(__file__).parent
SCRIPT_PATH = SCRIPT_DIR / "01_model_builder.py"

# ES: Configurar variables de entorno (backends no GUI)
# EN: Configure environment variables (non-GUI backends)
# JA: 環境変数を設定（非GUIバックエンド）
ENV = os.environ.copy()
ENV.setdefault("MPLBACKEND", "Agg")
ENV.setdefault("QT_QPA_PLATFORM", "offscreen")

# ES: Variables adicionales para evitar fragmentación del heap nativo
# EN: Additional variables to reduce native-heap fragmentation
# JA: ネイティブヒープ断片化を抑える追加設定
ENV["OMP_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["NUMEXPR_NUM_THREADS"] = "1"
ENV["MKL_SERVICE_FORCE_INTEL"] = "1"
ENV["OMP_DYNAMIC"] = "FALSE"
ENV["KMP_BLOCKTIME"] = "0"
ENV["KMP_AFFINITY"] = "disabled"

# ES: Número de folds (ajustar según configuración)
# EN: Number of folds (adjust per configuration)
# JA: fold数（設定に合わせて調整）
# ES: ★ IMPORTANTE: Cambiar este valor según Config.OUTER_SPLITS en tu configuración
# EN: ★ IMPORTANT: Set this value according to Config.OUTER_SPLITS in your config
# JA: ★ 重要：Config.OUTER_SPLITS に合わせて変更
NUM_FOLDS = 10  # Default value; adjust as needed

def main():
    """ES: Ejecuta cada fold en un proceso separado
    EN: Run each fold in a separate process
    JA: 各foldを別プロセスで実行
    """
    print("="*60)
    print("🚀 foldごとに別プロセスで解析を実行中")
    print(f"📊 fold数: {NUM_FOLDS}")
    print(f"📝 スクリプト: {SCRIPT_PATH}")
    print("="*60)
    
    for fold in range(NUM_FOLDS):
        print(f"\n{'='*60}")
        print(f"==> Fold 起動 {fold + 1}/{NUM_FOLDS}（index {fold}）")
        print(f"{'='*60}")
        
        try:
            # ES: Ejecutar fold en proceso separado
            # EN: Run fold in a separate process
            # JA: foldを別プロセスで実行
            # -u: unbuffered output (real-time logs)
            result = subprocess.run(
                [sys.executable, "-u", str(SCRIPT_PATH), "--single-outer-fold", str(fold)],
                env=ENV,
                check=True,  # Raise if it fails
                cwd=str(SCRIPT_DIR)  # Run from the script directory
            )
            
            print(f"✅ Fold {fold + 1} が完了しました")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Fold {fold + 1} が失敗しました（終了コード {e.returncode}）")
            print(f"⚠️  残りの fold 実行を停止します")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n⚠️  手動中断を検出しました")
            print(f"⚠️  Fold {fold + 1} が中断されました")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ すべての fold が完了しました")
    print("="*60)

if __name__ == "__main__":
    main()

