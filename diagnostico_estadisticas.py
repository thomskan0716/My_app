"""
Script de diagnóstico para verificar el problema de estadísticas de clasificación
Ejecutar con: python diagnostico_estadisticas.py <ruta_carpeta_analisis>
"""
import os
import re
import sys
from pathlib import Path

def diagnosticar_estadisticas(output_folder):
    """
    Diagnostica el problema con las estadísticas de clasificación
    """
    print("=" * 80)
    print("DIAGNÓSTICO DE ESTADÍSTICAS DE CLASIFICACIÓN")
    print("=" * 80)
    
    print(f"\n📁 Carpeta de salida: {output_folder}")
    if not os.path.exists(output_folder):
        print("❌ La carpeta de salida no existe")
        return False
    
    # Verificar estructura de carpetas
    print("\n📂 Verificando estructura de carpetas:")
    expected_paths = [
        ('02_本学習結果', os.path.join(output_folder, '02_本学習結果')),
        ('04_診断情報', os.path.join(output_folder, '02_本学習結果', '04_診断情報')),
        ('02_評価結果', os.path.join(output_folder, '02_本学習結果', '02_評価結果')),
    ]
    
    estructura_ok = True
    for name, path in expected_paths:
        exists = os.path.exists(path)
        is_dir = os.path.isdir(path) if exists else False
        status = '✅' if exists and is_dir else '❌'
        print(f"   {status} {name}: {path}")
        if not exists or not is_dir:
            estructura_ok = False
    
    # Buscar diagnostic_report.txt
    print("\n📄 Buscando diagnostic_report.txt:")
    diagnostic_report_path = os.path.join(output_folder, '02_本学習結果', '04_診断情報', 'diagnostic_report.txt')
    alternative_path = os.path.join(output_folder, '02_本学習結果', '02_評価結果', 'diagnostic_report.txt')
    
    report_path = None
    if os.path.exists(diagnostic_report_path):
        report_path = diagnostic_report_path
        print(f"   ✅ Encontrado en: {diagnostic_report_path}")
    elif os.path.exists(alternative_path):
        report_path = alternative_path
        print(f"   ✅ Encontrado en (alternativa): {alternative_path}")
    else:
        print(f"   ❌ No encontrado en ubicaciones esperadas")
        print(f"      Buscado en:")
        print(f"        1. {diagnostic_report_path}")
        print(f"        2. {alternative_path}")
        
        # Buscar recursivamente
        print(f"\n   🔍 Búsqueda recursiva de diagnostic_report.txt:")
        encontrado = False
        for root, dirs, files in os.walk(output_folder):
            if 'diagnostic_report.txt' in files:
                found_path = os.path.join(root, 'diagnostic_report.txt')
                print(f"      ✅ Encontrado en: {found_path}")
                report_path = found_path
                encontrado = True
                break
        
        if not encontrado:
            print(f"      ❌ No encontrado en ninguna ubicación")
            return False
    
    # Leer y analizar el contenido
    print(f"\n📖 Analizando archivo: {report_path}")
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"   ✅ Archivo leído correctamente")
        print(f"   Tamaño: {len(content)} caracteres")
        print(f"   Líneas: {len(content.splitlines())} líneas")
        
        # Mostrar primeras líneas
        print(f"\n📋 Primeras 20 líneas del archivo:")
        lines = content.splitlines()
        for i, line in enumerate(lines[:20], 1):
            print(f"   {i:3d}: {line}")
        
        # Probar expresiones regulares (igual que en 0sec.py)
        print(f"\n🔍 Probando expresiones regulares (igual que en 0sec.py):")
        diagnostic_data = {}
        problemas = []
        
        # [設定情報]
        print(f"\n   [設定情報]:")
        np_alpha_match = re.search(r'NP_ALPHA:\s*([\d.]+)', content)
        if np_alpha_match:
            diagnostic_data['np_alpha'] = np_alpha_match.group(1)
            print(f"      ✅ NP_ALPHA: {diagnostic_data['np_alpha']}")
        else:
            print(f"      ❌ NP_ALPHA: No encontrado")
            alt_match = re.search(r'NP_ALPHA[:\s]+([\d.]+)', content, re.IGNORECASE)
            if alt_match:
                diagnostic_data['np_alpha'] = alt_match.group(1)
                print(f"         ⚠️  Variación encontrada: {alt_match.group(1)}")
            else:
                problemas.append("NP_ALPHA no encontrado")
        
        objective_match = re.search(r'目的変数:\s*(.+)', content)
        if objective_match:
            diagnostic_data['objective'] = objective_match.group(1).strip()
            print(f"      ✅ 目的変数: {diagnostic_data['objective']}")
        else:
            print(f"      ❌ 目的変数: No encontrado")
            alt_match = re.search(r'目的変数[:\s]+(.+)', content)
            if alt_match:
                diagnostic_data['objective'] = alt_match.group(1).strip()
                print(f"         ⚠️  Variación encontrada: {alt_match.group(1).strip()}")
            else:
                problemas.append("目的変数 no encontrado")
        
        # [モデル情報]
        print(f"\n   [モデル情報]:")
        calibrator_match = re.search(r'Calibrator:\s*(.+)', content)
        if calibrator_match:
            diagnostic_data['calibrator'] = calibrator_match.group(1).strip()
            print(f"      ✅ Calibrator: {diagnostic_data['calibrator']}")
        else:
            print(f"      ❌ Calibrator: No encontrado")
            problemas.append("Calibrator no encontrado")
        
        # tau_pos con múltiples formatos
        tau_pos_match = re.search(r'τ\+\s*\(tau_pos\):\s*([\d.]+)', content)
        if not tau_pos_match:
            tau_pos_match = re.search(r'tau_pos[:\s]+([\d.]+)', content, re.IGNORECASE)
        if not tau_pos_match:
            tau_pos_match = re.search(r'τ\+[:\s]+([\d.]+)', content)
        if tau_pos_match:
            diagnostic_data['tau_pos'] = tau_pos_match.group(1)
            print(f"      ✅ τ+ (tau_pos): {diagnostic_data['tau_pos']}")
        else:
            print(f"      ❌ τ+ (tau_pos): No encontrado")
            problemas.append("τ+ (tau_pos) no encontrado")
        
        # tau_neg con múltiples formatos
        tau_neg_match = re.search(r'τ-\s*\(tau_neg\):\s*([\d.]+)', content)
        if not tau_neg_match:
            tau_neg_match = re.search(r'tau_neg[:\s]+([\d.]+)', content, re.IGNORECASE)
        if not tau_neg_match:
            tau_neg_match = re.search(r'τ-[:\s]+([\d.]+)', content)
        if tau_neg_match:
            diagnostic_data['tau_neg'] = tau_neg_match.group(1)
            print(f"      ✅ τ- (tau_neg): {diagnostic_data['tau_neg']}")
        else:
            print(f"      ❌ τ- (tau_neg): No encontrado")
            problemas.append("τ- (tau_neg) no encontrado")
        
        features_match = re.search(r'選択特徴量数:\s*(\d+)', content)
        if features_match:
            diagnostic_data['selected_features'] = features_match.group(1)
            print(f"      ✅ 選択特徴量数: {diagnostic_data['selected_features']}")
        else:
            print(f"      ❌ 選択特徴量数: No encontrado")
        
        # [予測結果統計]
        print(f"\n   [予測結果統計]:")
        total_data_match = re.search(r'総データ数:\s*([\d,]+)', content)
        if total_data_match:
            diagnostic_data['total_data'] = total_data_match.group(1).replace(',', '')
            print(f"      ✅ 総データ数: {diagnostic_data['total_data']}")
        else:
            print(f"      ❌ 総データ数: No encontrado")
        
        coverage_match = re.search(r'カバレッジ:\s*([\d.]+)%', content)
        if not coverage_match:
            coverage_match = re.search(r'カバレッジ[:\s]+([\d.]+)', content)
        if coverage_match:
            diagnostic_data['coverage'] = coverage_match.group(1)
            print(f"      ✅ カバレッジ: {diagnostic_data['coverage']}%")
        else:
            print(f"      ❌ カバレッジ: No encontrado")
        
        # [ノイズ付加設定]
        print(f"\n   [ノイズ付加設定]:")
        noise_enabled_match = re.search(r'ノイズ付加:\s*(True|False)', content)
        if noise_enabled_match:
            diagnostic_data['noise_enabled'] = noise_enabled_match.group(1) == 'True'
            print(f"      ✅ ノイズ付加: {diagnostic_data['noise_enabled']}")
        else:
            print(f"      ❌ ノイズ付加: No encontrado")
        
        noise_level_match = re.search(r'ノイズレベル:\s*([\d.]+)\s*ppm', content)
        if not noise_level_match:
            noise_level_match = re.search(r'ノイズレベル[:\s]+([\d.]+)', content)
        if noise_level_match:
            diagnostic_data['noise_level'] = noise_level_match.group(1)
            print(f"      ✅ ノイズレベル: {diagnostic_data['noise_level']} ppm")
        else:
            print(f"      ❌ ノイズレベル: No encontrado")
        
        # Resumen de datos parseados
        print(f"\n📊 Resumen de datos parseados:")
        print(f"   Total de campos encontrados: {len(diagnostic_data)}")
        for key, value in diagnostic_data.items():
            print(f"      - {key}: {value}")
        
        # Verificar qué se mostraría en la UI
        print(f"\n🎨 Análisis de qué se mostraría en la UI:")
        has_basic_info = any([
            diagnostic_data.get('objective'),
            diagnostic_data.get('np_alpha'),
            diagnostic_data.get('total_data'),
            diagnostic_data.get('coverage'),
            diagnostic_data.get('selected_features'),
        ])
        print(f"   Información básica (info_label): {'✅ Sí' if has_basic_info else '❌ No'}")
        if has_basic_info:
            print(f"      Se mostraría: 解析完了時刻, 解析時間")
            if diagnostic_data.get('objective'):
                print(f"      + 目的変数: {diagnostic_data['objective']}")
            if diagnostic_data.get('np_alpha'):
                print(f"      + NP_ALPHA: {diagnostic_data['np_alpha']}")
            if diagnostic_data.get('total_data'):
                print(f"      + 総データ数: {diagnostic_data['total_data']}")
            if diagnostic_data.get('coverage'):
                print(f"      + カバレッジ: {diagnostic_data['coverage']}%")
            if diagnostic_data.get('selected_features'):
                print(f"      + 選択特徴量数: {diagnostic_data['selected_features']}")
        
        has_model_info = diagnostic_data.get('tau_pos') and diagnostic_data.get('tau_neg')
        print(f"   Información del modelo (metric_card): {'✅ Sí' if has_model_info else '❌ No'}")
        if not has_model_info:
            print(f"      ⚠️  REQUIERE: tau_pos Y tau_neg (ambos)")
            print(f"      tau_pos encontrado: {bool(diagnostic_data.get('tau_pos'))}")
            print(f"      tau_neg encontrado: {bool(diagnostic_data.get('tau_neg'))}")
        
        has_noise_info = diagnostic_data.get('noise_enabled')
        print(f"   Información de ruido (noise_card): {'✅ Sí' if has_noise_info else '❌ No'}")
        
        # Diagnóstico del problema
        print(f"\n🔍 DIAGNÓSTICO DEL PROBLEMA:")
        if not has_basic_info and not has_model_info and not has_noise_info:
            print(f"   ❌ PROBLEMA CRÍTICO: No se encontraron datos suficientes para mostrar")
            print(f"      Esto explicaría por qué la pantalla está vacía")
            print(f"      Problemas detectados: {len(problemas)}")
            for p in problemas:
                print(f"         - {p}")
            return False
        elif not has_model_info:
            print(f"   ⚠️  PROBLEMA: Falta información del modelo (tau_pos o tau_neg)")
            print(f"      Se mostrará información básica pero no la sección de modelo")
        else:
            print(f"   ✅ Todo parece estar bien. Los datos deberían mostrarse correctamente.")
            print(f"      Si aún no se ven, el problema puede estar en:")
            print(f"        1. El layout no se está actualizando correctamente")
            print(f"        2. Los widgets se crean pero no son visibles (problema de CSS)")
            print(f"        3. El container_layout no está conectado correctamente")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error leyendo archivo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_folder = sys.argv[1]
    else:
        # Buscar la última carpeta de análisis de clasificación
        script_dir = Path(__file__).parent
        base_folder = script_dir / "05_分類"
        if not base_folder.exists():
            # Intentar desde el directorio actual
            base_folder = Path("05_分類")
        
        if base_folder.exists():
            # Buscar la carpeta más reciente
            folders = [f for f in base_folder.iterdir() if f.is_dir() and "分類解析結果" in f.name]
            if folders:
                # Ordenar por fecha de modificación
                folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                output_folder = str(folders[0])
                print(f"📁 Usando carpeta más reciente: {output_folder}")
            else:
                print("❌ No se encontraron carpetas de análisis")
                print("   Uso: python diagnostico_estadisticas.py <ruta_carpeta_salida>")
                sys.exit(1)
        else:
            print("❌ No se encontró la carpeta base 05_分類")
            print("   Uso: python diagnostico_estadisticas.py <ruta_carpeta_salida>")
            sys.exit(1)
    
    success = diagnosticar_estadisticas(output_folder)
    sys.exit(0 if success else 1)














