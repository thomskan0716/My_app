"""
ES: Módulo de debug para detectar DLLs cargadas, especialmente OpenMP runtimes.
EN: Debug module to detect loaded DLLs, especially OpenMP runtimes.
JA: 読み込まれたDLL（特にOpenMPランタイム）を検出するデバッグモジュール。

ES: Útil para diagnosticar conflictos de DLLs en Windows.
EN: Useful for diagnosing DLL conflicts on Windows.
JA: WindowsでのDLL競合の診断に有用。
"""
import sys
import os

def get_loaded_dlls():
    """
    ES: Obtiene lista de DLLs cargadas en el proceso actual (Windows).
    EN: Get the list of DLLs loaded in the current process (Windows).
    JA: 現在のプロセスで読み込まれているDLL一覧を取得（Windows）。

    ES: Usa múltiples métodos para asegurar que funcione.
    EN: Uses multiple methods to improve reliability.
    JA: 動作保証のため複数の手法を使用。
    
    Returns
    -------
    list
        ES: Lista de nombres de DLLs cargadas
        EN: List of loaded DLL names
        JA: 読み込まれたDLL名のリスト
    """
    if sys.platform != 'win32':
        return []
    
    dll_names = []
    
    # ES: Método 1: EnumProcessModules (más completo)
    # EN: Method 1: EnumProcessModules (most complete)
    # JA: 方法1：EnumProcessModules（最も網羅的）
    try:
        import ctypes
        from ctypes import wintypes
        
        # ES: EnumProcessModules requiere psapi.dll
        # EN: EnumProcessModules requires psapi.dll
        # JA: EnumProcessModules は psapi.dll が必要
        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        
        # ES: Obtener handle del proceso actual
        # EN: Get current process handle
        # JA: 現在プロセスのハンドルを取得
        h_process = kernel32.GetCurrentProcess()
        
        # ES: Buffer para módulos (aumentado a 2048 para procesos con muchas DLLs)
        # EN: Module buffer (increased to 2048 for processes with many DLLs)
        # JA: モジュールバッファ（DLLが多いプロセス向けに2048へ拡大）
        module_handles = (wintypes.HMODULE * 2048)()
        cb_needed = wintypes.DWORD()
        
        # ES: Enumerar módulos
        # EN: Enumerate modules
        # JA: モジュールを列挙
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
                        if dll_path:  # Ensure not empty
                            dll_name = os.path.basename(dll_path).lower()
                            if dll_name:
                                dll_names.append(dll_name)
                except:
                    continue  # Continue with next module on failure
            
            if dll_names:
                return sorted(set(dll_names))  # De-duplicate and sort
    except Exception as e:
        pass  # Try fallback method
    
    # ES: Método 2: Usar ctypes.util.find_library y sys.modules (fallback)
    # EN: Method 2: Use ctypes.util.find_library and sys.modules (fallback)
    # JA: 方法2：ctypes.util.find_library と sys.modules を使用（フォールバック）
    try:
        import ctypes.util
        # sys is already imported at the top; do not re-import
        
        # ES: Buscar DLLs conocidas en sys.modules
        # EN: Look for known DLLs in sys.modules
        # JA: sys.modules から既知DLLを探索
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
    
    # ES: Si aún no tenemos DLLs, retornar lista vacía (pero no None)
    # EN: If still no DLLs, return an empty list (not None)
    # JA: それでもDLLが無ければ空リストを返す（Noneではない）
    return sorted(set(dll_names)) if dll_names else []


def detect_openmp_runtimes(dll_list=None):
    """
    ES: Detecta qué runtimes de OpenMP están cargados.
    EN: Detect which OpenMP runtimes are loaded.
    JA: 読み込まれているOpenMPランタイムを検出。
    
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
    
    # ES: Patrones de nombres de DLLs OpenMP
    # EN: OpenMP DLL name patterns
    # JP: OpenMP DLL名のパターン
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
        
        # ES: Buscar en cada categoría
        # EN: Search in each category
        # JP: 各カテゴリで検索
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
        
        # ES: Si no está en ninguna categoría conocida pero contiene "omp"
        # EN: If it's not in any known category but contains "omp"
        # JP: 既知カテゴリに該当しないが「omp」を含む場合
        if not found and 'omp' in dll_lower:
            detected['other'].append(dll)
            detected['all_omp_dlls'].append(dll)
    
    return detected


def detect_blas_libraries(dll_list=None):
    """
    ES: Detecta qué librerías BLAS/LAPACK están cargadas.
    EN: Detect which BLAS/LAPACK libraries are loaded.
    JA: 読み込まれているBLAS/LAPACKライブラリを検出。
    
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
    ES: Detecta DLLs relacionadas con GUI (Qt, Tkinter, etc.).
    EN: Detect GUI-related DLLs (Qt, Tkinter, etc.).
    JA: GUI関連DLL（Qt/Tkinter等）を検出。
    
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
    print(f"[DLL] DLLレポート - ステージ: {stage}", flush=True)
    print("="*80, flush=True)
    
    try:
        dll_list = get_loaded_dlls()
        
        if not dll_list:
            print("[警告] 読み込まれたDLLを取得できませんでした（権限が必要な場合があります）", flush=True)
            print("="*80, flush=True)
            return
        
        print(f"\n[INFO] Total de DLLs cargadas: {len(dll_list)}", flush=True)
        
        # Mostrar algunas DLLs relevantes (primeras 20)
        relevant_dlls = [dll for dll in dll_list if any(keyword in dll for keyword in 
            ['omp', 'mkl', 'blas', 'openblas', 'qt', 'tk', 'python', 'numpy', 'sklearn'])]
        if relevant_dlls:
            print(f"\n[情報] 関連DLLを検出しました（{len(relevant_dlls)}件）:", flush=True)
            for dll in relevant_dlls[:20]:  # Mostrar máximo 20
                print(f"     - {dll}", flush=True)
            if len(relevant_dlls) > 20:
                print(f"     ... ほか {len(relevant_dlls) - 20} 件", flush=True)
        
        # Detectar OpenMP
        omp_info = detect_openmp_runtimes(dll_list)
        print(f"\n[OMP] OpenMPランタイム検出結果:", flush=True)
        if omp_info['all_omp_dlls']:
            print(f"  [警告] 複数のOpenMPランタイム: {len(omp_info['all_omp_dlls'])}個のDLLを検出", flush=True)
            if omp_info['intel']:
                print(f"     - Intel OpenMP: {', '.join(omp_info['intel'][:5])}", flush=True)
                if len(omp_info['intel']) > 5:
                    print(f"       ... ほか {len(omp_info['intel']) - 5} 件", flush=True)
            if omp_info['msvc']:
                print(f"     - MSVC OpenMP: {', '.join(omp_info['msvc'][:5])}", flush=True)
                if len(omp_info['msvc']) > 5:
                    print(f"       ... ほか {len(omp_info['msvc']) - 5} 件", flush=True)
            if omp_info['gcc']:
                print(f"     - GCC OpenMP: {', '.join(omp_info['gcc'][:5])}", flush=True)
                if len(omp_info['gcc']) > 5:
                    print(f"       ... ほか {len(omp_info['gcc']) - 5} 件", flush=True)
            if omp_info['other']:
                print(f"     - その他のOpenMP: {', '.join(omp_info['other'][:5])}", flush=True)
                if len(omp_info['other']) > 5:
                    print(f"       ... ほか {len(omp_info['other']) - 5} 件", flush=True)
            
            # ES: ADVERTENCIA si hay múltiples runtimes
            # EN: WARNING if there are multiple runtimes
            # JP: 警告: 複数のランタイムがある場合
            total_runtimes = sum([
                len(omp_info['intel']) > 0,
                len(omp_info['msvc']) > 0,
                len(omp_info['gcc']) > 0,
                len(omp_info['other']) > 0
            ])
            if total_runtimes > 1:
                print(f"\n  [エラー] 競合を検出: 複数のOpenMPランタイムが同時に読み込まれています！", flush=True)
                print(f"     これはヒープ破損 (0xC0000374) を引き起こす可能性があります", flush=True)
            else:
                print(f"  [OK] OpenMPランタイムは1つのみ（競合なし）", flush=True)
        else:
            print("  [OK] OpenMP DLLは検出されませんでした", flush=True)
        
        # Detectar BLAS
        blas_info = detect_blas_libraries(dll_list)
        print(f"\n[BLAS] BLAS/LAPACKライブラリ検出結果:", flush=True)
        if blas_info['all_blas_dlls']:
            if blas_info['mkl']:
                print(f"  - Intel MKL: {len(blas_info['mkl'])} DLLs", flush=True)
            if blas_info['openblas']:
                print(f"  - OpenBLAS: {', '.join(blas_info['openblas'][:5])}", flush=True)
            if blas_info['blas']:
                print(f"  - 汎用BLAS: {', '.join(blas_info['blas'][:5])}", flush=True)
            if blas_info['lapack']:
                print(f"  - LAPACK: {', '.join(blas_info['lapack'][:5])}", flush=True)
        else:
            print("  [OK] BLAS/LAPACK DLLは検出されませんでした", flush=True)
        
        # Detectar GUI
        gui_info = detect_gui_dlls(dll_list)
        print(f"\n[GUI] GUI/COM DLL検出結果:", flush=True)
        if gui_info['all_gui_dlls']:
            if gui_info['qt']:
                print(f"  [警告] Qt: {len(gui_info['qt'])} DLL（OpenMPを含む可能性があります）", flush=True)
            if gui_info['tkinter']:
                print(f"  [WARN] Tkinter: {', '.join(gui_info['tkinter'][:5])}", flush=True)
            if gui_info['matplotlib']:
                print(f"  - Matplotlib (freetype/png): {len(gui_info['matplotlib'])} DLLs", flush=True)
            if gui_info['com']:
                print(f"  [WARN] COM: {', '.join(gui_info['com'][:5])}", flush=True)
        else:
            print("  [OK] GUI/COM DLLは検出されませんでした", flush=True)
        
        # Variables de entorno relevantes
        print(f"\n[ENV] 関連環境変数:", flush=True)
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
            value = os.environ.get(var, '未設定')
            print(f"  {var} = {value}", flush=True)
        
    except Exception as e:
        print(f"\n[エラー] DLLレポート生成中にエラー: {e}", flush=True)
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
        
        # ES: Verificar múltiples runtimes OpenMP
        # EN: Check for multiple OpenMP runtimes
        # JP: 複数のOpenMPランタイムを確認する
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
                f"複数のOpenMPランタイムを検出: "
                f"Intel={len(omp_info['intel'])}, "
                f"MSVC={len(omp_info['msvc'])}, "
                f"GCC={len(omp_info['gcc'])}, "
                f"その他={len(omp_info['other'])}"
            )
            print(f"[エラー] 競合を検出: {conflicts['details'][0]}", flush=True)
        else:
            print(f"[OK] DLLの競合は検出されませんでした", flush=True)
        
        return conflicts
    except Exception as e:
        print(f"[エラー] DLL競合の確認中にエラー: {e}", flush=True)
        return {'has_conflict': False, 'conflict_type': None, 'details': []}
