"""
Módulo de debug para detectar DLLs cargadas, especialmente OpenMP runtimes
Útil para diagnosticar conflictos de DLLs en Windows
"""
import sys
import os

def get_loaded_dlls():
    """
    Obtiene lista de DLLs cargadas en el proceso actual (Windows)
    Usa múltiples métodos para asegurar que funcione
    
    Returns
    -------
    list
        Lista de nombres de DLLs cargadas
    """
    if sys.platform != 'win32':
        return []
    
    dll_names = []
    
    # Método 1: EnumProcessModules (más completo)
    try:
        import ctypes
        from ctypes import wintypes
        
        # EnumProcessModules requiere psapi.dll
        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        
        # Obtener handle del proceso actual
        h_process = kernel32.GetCurrentProcess()
        
        # Buffer para módulos (aumentado a 2048 para procesos con muchas DLLs)
        module_handles = (wintypes.HMODULE * 2048)()
        cb_needed = wintypes.DWORD()
        
        # Enumerar módulos
        if psapi.EnumProcessModules(
            h_process,
            ctypes.byref(module_handles),
            ctypes.sizeof(module_handles),
            ctypes.byref(cb_needed)
        ):
            num_modules = min(cb_needed.value // ctypes.sizeof(wintypes.HMODULE), 2048)
            
            for i in range(num_modules):
                try:
                    module_name = ctypes.create_unicode_buffer(260)  # MAX_PATH
                    if psapi.GetModuleFileNameExW(
                        h_process,
                        module_handles[i],
                        module_name,
                        260
                    ):
                        dll_path = module_name.value
                        if dll_path:  # Verificar que no esté vacío
                            dll_name = os.path.basename(dll_path).lower()
                            if dll_name:
                                dll_names.append(dll_name)
                except:
                    continue  # Continuar con el siguiente módulo si falla
            
            if dll_names:
                return sorted(set(dll_names))  # Eliminar duplicados y ordenar
    except Exception as e:
        pass  # Intentar método alternativo
    
    # Método 2: Usar ctypes.util.find_library y sys.modules (fallback)
    try:
        import ctypes.util
        # sys ya está importado al inicio del archivo, no reimportar
        
        # Buscar DLLs conocidas en sys.modules
        known_dlls = []
        for module_name, module_obj in sys.modules.items():
            if hasattr(module_obj, '__file__'):
                try:
                    file_path = module_obj.__file__
                    if file_path and (file_path.endswith('.dll') or file_path.endswith('.pyd')):
                        dll_name = os.path.basename(file_path).lower()
                        known_dlls.append(dll_name)
                except:
                    continue
        
        if known_dlls:
            dll_names.extend(known_dlls)
    except Exception as e:
        pass
    
    # Si aún no tenemos DLLs, retornar lista vacía (pero no None)
    return sorted(set(dll_names)) if dll_names else []


def detect_openmp_runtimes(dll_list=None):
    """
    Detecta qué runtimes de OpenMP están cargados
    
    Parameters
    ----------
    dll_list : list, optional
        Lista de DLLs. Si es None, se obtiene automáticamente.
    
    Returns
    -------
    dict
        Diccionario con información de runtimes OpenMP detectados
    """
    if dll_list is None:
        dll_list = get_loaded_dlls()
    
    # Patrones de nombres de DLLs OpenMP
    omp_patterns = {
        'intel': ['libiomp5md.dll', 'libiomp5.dll', 'iomp5md.dll'],
        'msvc': ['vcomp140.dll', 'vcomp120.dll', 'vcomp110.dll', 'vcomp100.dll'],
        'gcc': ['libgomp-1.dll', 'libgomp.dll'],
        'other': []
    }
    
    detected = {
        'intel': [],
        'msvc': [],
        'gcc': [],
        'other': [],
        'all_omp_dlls': []
    }
    
    for dll in dll_list:
        dll_lower = dll.lower()
        
        # Buscar en cada categoría
        found = False
        for category, patterns in omp_patterns.items():
            for pattern in patterns:
                if pattern in dll_lower:
                    detected[category].append(dll)
                    detected['all_omp_dlls'].append(dll)
                    found = True
                    break
            if found:
                break
        
        # Si no está en ninguna categoría conocida pero contiene "omp"
        if not found and 'omp' in dll_lower:
            detected['other'].append(dll)
            detected['all_omp_dlls'].append(dll)
    
    return detected


def detect_blas_libraries(dll_list=None):
    """
    Detecta qué librerías BLAS/LAPACK están cargadas
    
    Parameters
    ----------
    dll_list : list, optional
        Lista de DLLs. Si es None, se obtiene automáticamente.
    
    Returns
    -------
    dict
        Diccionario con información de librerías BLAS detectadas
    """
    if dll_list is None:
        dll_list = get_loaded_dlls()
    
    blas_patterns = {
        'mkl': ['mkl_', 'libmkl'],
        'openblas': ['openblas', 'libopenblas'],
        'blas': ['blas', 'libblas'],
        'lapack': ['lapack', 'liblapack']
    }
    
    detected = {
        'mkl': [],
        'openblas': [],
        'blas': [],
        'lapack': [],
        'all_blas_dlls': []
    }
    
    for dll in dll_list:
        dll_lower = dll.lower()
        
        for category, patterns in blas_patterns.items():
            for pattern in patterns:
                if pattern in dll_lower:
                    detected[category].append(dll)
                    detected['all_blas_dlls'].append(dll)
                    break
    
    return detected


def detect_gui_dlls(dll_list=None):
    """
    Detecta DLLs relacionadas con GUI (Qt, Tkinter, etc.)
    
    Parameters
    ----------
    dll_list : list, optional
        Lista de DLLs. Si es None, se obtiene automáticamente.
    
    Returns
    -------
    dict
        Diccionario con información de DLLs GUI detectadas
    """
    if dll_list is None:
        dll_list = get_loaded_dlls()
    
    gui_patterns = {
        'qt': ['qt', 'qwindows', 'qminimal'],
        'tkinter': ['tcl', 'tk'],
        'matplotlib': ['freetype', 'png', 'zlib'],
        'com': ['pythoncom', 'pywintypes', 'oleaut32', 'ole32']
    }
    
    detected = {
        'qt': [],
        'tkinter': [],
        'matplotlib': [],
        'com': [],
        'all_gui_dlls': []
    }
    
    for dll in dll_list:
        dll_lower = dll.lower()
        
        for category, patterns in gui_patterns.items():
            for pattern in patterns:
                if pattern in dll_lower:
                    detected[category].append(dll)
                    detected['all_gui_dlls'].append(dll)
                    break
    
    return detected


def print_dll_report(stage="Unknown"):
    """
    Imprime un reporte completo de DLLs cargadas
    
    Parameters
    ----------
    stage : str
        Etapa del proceso donde se genera el reporte (para debug)
    """
    print("\n" + "="*80, flush=True)
    print(f"[DLL] REPORTE DE DLLs - Etapa: {stage}", flush=True)
    print("="*80, flush=True)
    
    try:
        dll_list = get_loaded_dlls()
        
        if not dll_list:
            print("[WARN] No se pudieron obtener las DLLs cargadas (puede requerir permisos)", flush=True)
            print("="*80, flush=True)
            return
        
        print(f"\n[INFO] Total de DLLs cargadas: {len(dll_list)}", flush=True)
        
        # Mostrar algunas DLLs relevantes (primeras 20)
        relevant_dlls = [dll for dll in dll_list if any(keyword in dll for keyword in 
            ['omp', 'mkl', 'blas', 'openblas', 'qt', 'tk', 'python', 'numpy', 'sklearn'])]
        if relevant_dlls:
            print(f"\n[INFO] DLLs relevantes detectadas ({len(relevant_dlls)}):", flush=True)
            for dll in relevant_dlls[:20]:  # Mostrar máximo 20
                print(f"     - {dll}", flush=True)
            if len(relevant_dlls) > 20:
                print(f"     ... y {len(relevant_dlls) - 20} mas", flush=True)
        
        # Detectar OpenMP
        omp_info = detect_openmp_runtimes(dll_list)
        print(f"\n[OMP] RUNTIMES OPENMP DETECTADOS:", flush=True)
        if omp_info['all_omp_dlls']:
            print(f"  [WARN] MULTIPLES RUNTIMES OPENMP: {len(omp_info['all_omp_dlls'])} DLLs encontradas", flush=True)
            if omp_info['intel']:
                print(f"     - Intel OpenMP: {', '.join(omp_info['intel'][:5])}", flush=True)
                if len(omp_info['intel']) > 5:
                    print(f"       ... y {len(omp_info['intel']) - 5} mas", flush=True)
            if omp_info['msvc']:
                print(f"     - MSVC OpenMP: {', '.join(omp_info['msvc'][:5])}", flush=True)
                if len(omp_info['msvc']) > 5:
                    print(f"       ... y {len(omp_info['msvc']) - 5} mas", flush=True)
            if omp_info['gcc']:
                print(f"     - GCC OpenMP: {', '.join(omp_info['gcc'][:5])}", flush=True)
                if len(omp_info['gcc']) > 5:
                    print(f"       ... y {len(omp_info['gcc']) - 5} mas", flush=True)
            if omp_info['other']:
                print(f"     - Otros OpenMP: {', '.join(omp_info['other'][:5])}", flush=True)
                if len(omp_info['other']) > 5:
                    print(f"       ... y {len(omp_info['other']) - 5} mas", flush=True)
            
            # ADVERTENCIA si hay múltiples runtimes
            total_runtimes = sum([
                len(omp_info['intel']) > 0,
                len(omp_info['msvc']) > 0,
                len(omp_info['gcc']) > 0,
                len(omp_info['other']) > 0
            ])
            if total_runtimes > 1:
                print(f"\n  [ERROR] CONFLICTO DETECTADO: Multiples runtimes OpenMP cargados simultaneamente!", flush=True)
                print(f"     Esto puede causar heap corruption (0xC0000374)", flush=True)
            else:
                print(f"  [OK] Solo un runtime OpenMP detectado (sin conflicto)", flush=True)
        else:
            print("  [OK] No se detectaron DLLs OpenMP", flush=True)
        
        # Detectar BLAS
        blas_info = detect_blas_libraries(dll_list)
        print(f"\n[BLAS] LIBRERIAS BLAS/LAPACK DETECTADAS:", flush=True)
        if blas_info['all_blas_dlls']:
            if blas_info['mkl']:
                print(f"  - Intel MKL: {len(blas_info['mkl'])} DLLs", flush=True)
            if blas_info['openblas']:
                print(f"  - OpenBLAS: {', '.join(blas_info['openblas'][:5])}", flush=True)
            if blas_info['blas']:
                print(f"  - BLAS generico: {', '.join(blas_info['blas'][:5])}", flush=True)
            if blas_info['lapack']:
                print(f"  - LAPACK: {', '.join(blas_info['lapack'][:5])}", flush=True)
        else:
            print("  [OK] No se detectaron DLLs BLAS/LAPACK", flush=True)
        
        # Detectar GUI
        gui_info = detect_gui_dlls(dll_list)
        print(f"\n[GUI] DLLs GUI/COM DETECTADAS:", flush=True)
        if gui_info['all_gui_dlls']:
            if gui_info['qt']:
                print(f"  [WARN] Qt: {len(gui_info['qt'])} DLLs (pueden traer OpenMP)", flush=True)
            if gui_info['tkinter']:
                print(f"  [WARN] Tkinter: {', '.join(gui_info['tkinter'][:5])}", flush=True)
            if gui_info['matplotlib']:
                print(f"  - Matplotlib (freetype/png): {len(gui_info['matplotlib'])} DLLs", flush=True)
            if gui_info['com']:
                print(f"  [WARN] COM: {', '.join(gui_info['com'][:5])}", flush=True)
        else:
            print("  [OK] No se detectaron DLLs GUI/COM", flush=True)
        
        # Variables de entorno relevantes
        print(f"\n[ENV] VARIABLES DE ENTORNO RELEVANTES:", flush=True)
        env_vars = [
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'MKL_THREADING_LAYER',
            'MKL_SERVICE_FORCE_INTEL',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'MPLBACKEND',
            'QT_QPA_PLATFORM',
            'KMP_DUPLICATE_LIB_OK',
            'PYTHONIOENCODING'
        ]
        for var in env_vars:
            value = os.environ.get(var, 'NO DEFINIDA')
            print(f"  {var} = {value}", flush=True)
        
    except Exception as e:
        print(f"\n[ERROR] ERROR al generar reporte de DLLs: {e}", flush=True)
        import traceback
        print(f"   Traceback: {traceback.format_exc()}", flush=True)
    
    print("="*80 + "\n", flush=True)


def check_dll_conflicts():
    """
    Verifica si hay conflictos potenciales de DLLs
    
    Returns
    -------
    dict
        Diccionario con información de conflictos detectados
    """
    try:
        dll_list = get_loaded_dlls()
        omp_info = detect_openmp_runtimes(dll_list)
        
        conflicts = {
            'has_conflict': False,
            'conflict_type': None,
            'details': []
        }
        
        # Verificar múltiples runtimes OpenMP
        total_omp_runtimes = sum([
            len(omp_info['intel']) > 0,
            len(omp_info['msvc']) > 0,
            len(omp_info['gcc']) > 0,
            len(omp_info['other']) > 0
        ])
        
        if total_omp_runtimes > 1:
            conflicts['has_conflict'] = True
            conflicts['conflict_type'] = 'multiple_openmp_runtimes'
            conflicts['details'].append(
                f"Multiples runtimes OpenMP detectados: "
                f"Intel={len(omp_info['intel'])}, "
                f"MSVC={len(omp_info['msvc'])}, "
                f"GCC={len(omp_info['gcc'])}, "
                f"Otros={len(omp_info['other'])}"
            )
            print(f"[ERROR] CONFLICTO DETECTADO: {conflicts['details'][0]}", flush=True)
        else:
            print(f"[OK] No se detectaron conflictos de DLLs", flush=True)
        
        return conflicts
    except Exception as e:
        print(f"[ERROR] Error al verificar conflictos de DLLs: {e}", flush=True)
        return {'has_conflict': False, 'conflict_type': None, 'details': []}
