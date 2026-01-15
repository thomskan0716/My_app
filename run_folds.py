"""
Launcher para ejecutar cada fold en un proceso separado
Esto evita acumulación de fragmentación del heap nativo entre folds
"""
import os
import sys
import subprocess
from pathlib import Path

# Obtener ruta del script desde el directorio actual
SCRIPT_DIR = Path(__file__).parent
SCRIPT_PATH = SCRIPT_DIR / "01_model_builder.py"

# Configurar variables de entorno (backends no GUI)
ENV = os.environ.copy()
ENV.setdefault("MPLBACKEND", "Agg")
ENV.setdefault("QT_QPA_PLATFORM", "offscreen")

# Variables adicionales para evitar fragmentación del heap nativo
ENV["OMP_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["NUMEXPR_NUM_THREADS"] = "1"
ENV["MKL_SERVICE_FORCE_INTEL"] = "1"
ENV["OMP_DYNAMIC"] = "FALSE"
ENV["KMP_BLOCKTIME"] = "0"
ENV["KMP_AFFINITY"] = "disabled"

# Número de folds (ajustar según configuración)
# ★ IMPORTANTE: Cambiar este valor según Config.OUTER_SPLITS en tu configuración
NUM_FOLDS = 10  # Valor por defecto, ajustar según necesidad

def main():
    """Ejecuta cada fold en un proceso separado"""
    print("="*60)
    print("🚀 Ejecutando análisis con un proceso por fold")
    print(f"📊 Total de folds: {NUM_FOLDS}")
    print(f"📝 Script: {SCRIPT_PATH}")
    print("="*60)
    
    for fold in range(NUM_FOLDS):
        print(f"\n{'='*60}")
        print(f"==> Lanzando Fold {fold + 1}/{NUM_FOLDS} (índice {fold})")
        print(f"{'='*60}")
        
        try:
            # Ejecutar fold en proceso separado
            # -u: unbuffered output (ver logs en tiempo real)
            result = subprocess.run(
                [sys.executable, "-u", str(SCRIPT_PATH), "--single-outer-fold", str(fold)],
                env=ENV,
                check=True,  # Lanza excepción si falla
                cwd=str(SCRIPT_DIR)  # Ejecutar desde el directorio del script
            )
            
            print(f"✅ Fold {fold + 1} completado exitosamente")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Fold {fold + 1} falló con código {e.returncode}")
            print(f"⚠️  Deteniendo ejecución de folds restantes")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupción manual detectada")
            print(f"⚠️  Fold {fold + 1} fue interrumpido")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ Todos los folds completados exitosamente")
    print("="*60)

if __name__ == "__main__":
    main()

