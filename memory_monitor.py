"""
ES: Sistema de monitoreo de memoria y fragmentación del heap.
EN: Memory and heap-fragmentation monitoring system.
JA: メモリとヒープ断片化のモニタリングシステム。

ES: Detecta qué está fragmentando el heap y visualiza en tiempo real.
EN: Detects what is fragmenting the heap and visualizes it in real time.
JA: 断片化要因を検出し、リアルタイムで可視化。
"""
import os
import sys
import time
import json
import gc
import threading
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import traceback

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil no disponible. Instalar con: pip install psutil")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class MemoryMonitor:
    """ES: Monitor de memoria y fragmentación del heap
    EN: Memory and heap-fragmentation monitor
    JA: メモリとヒープ断片化のモニター
    """
    
    def __init__(self, pid: Optional[int] = None, log_file: Optional[str] = None):
        """
        ES: Inicializa el monitor de memoria
        EN: Initialize the memory monitor
        JA: メモリモニターを初期化
        
        Parameters
        ----------
        pid : int, optional
            ES: PID del proceso a monitorear (None = proceso actual)
            EN: PID of the process to monitor (None = current process)
            JA: 監視対象プロセスのPID（Noneなら現在プロセス）
        log_file : str, optional
            ES: Archivo para guardar logs de memoria
            EN: File to write memory logs
            JA: メモリログの保存先ファイル
        """
        self.pid = pid or os.getpid()
        self.log_file = log_file
        self.monitoring = False
        self.monitor_thread = None
        
        # ES: Historial de métricas | EN: Metrics history | JA: メトリクス履歴
        self.memory_history = deque(maxlen=5000)  # ~2.5 horas a 2 segundos
        self.fragmentation_history = deque(maxlen=5000)
        self.object_counts = deque(maxlen=5000)
        self.gc_stats_history = deque(maxlen=5000)
        
        # ES: Tracking de objetos grandes | EN: Large-object tracking | JA: 大きいオブジェクトの追跡
        self.large_objects = []  # Lista de objetos > 10MB
        self.object_type_counts = defaultdict(int)
        self.allocation_events = []  # Large allocation events
        
        # ES: Métricas de fragmentación proxy | EN: Proxy fragmentation metrics | JA: 代理の断片化指標
        self.fragmentation_score = 0.0
        self.last_gc_time = 0.0
        self.gc_frequency = 0.0
        
        # ES: Estadísticas acumuladas | EN: Accumulated stats | JA: 累積統計
        self.stats = {
            'peak_memory': 0,
            'current_memory': 0,
            'total_allocations': 0,
            'total_deallocations': 0,
            'gc_collections': 0,
            'fragmentation_events': 0,
        }
        
    def start_monitoring(self, interval: float = 2.0):
        """
        ES: Inicia monitoreo en background
        EN: Start monitoring in the background
        JA: バックグラウンドで監視開始
        
        Parameters
        ----------
        interval : float
            ES: Intervalo de muestreo en segundos
            EN: Sampling interval in seconds
            JA: サンプリング間隔（秒）
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print(f"📊 Monitoreo de memoria iniciado (PID: {self.pid}, intervalo: {interval}s)")
    
    def stop_monitoring(self):
        """ES: Detiene el monitoreo
        EN: Stop monitoring
        JA: 監視を停止
        """
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("📊 Monitoreo de memoria detenido")
    
    def _monitor_loop(self, interval: float):
        """ES: Loop principal de monitoreo
        EN: Main monitoring loop
        JA: 監視のメインループ
        """
        while self.monitoring:
            try:
                self._collect_metrics()
                time.sleep(interval)
            except Exception as e:
                print(f"⚠️ Error en monitoreo: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self):
        """ES: Recolecta métricas de memoria
        EN: Collect memory metrics
        JA: メモリメトリクスを収集
        """
        timestamp = time.time()
        
        # Memoria del proceso
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(self.pid)
                mem_info = process.memory_info()
                self.stats['current_memory'] = mem_info.rss / 1024 / 1024  # MB
                self.stats['peak_memory'] = max(self.stats['peak_memory'], self.stats['current_memory'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Proceso terminado o sin acceso
                self.monitoring = False
                return
        else:
            # Fallback sin psutil
            try:
                import resource
                mem_info = resource.getrusage(resource.RUSAGE_SELF)
                self.stats['current_memory'] = mem_info.ru_maxrss / 1024  # KB to MB (Linux)
            except:
                self.stats['current_memory'] = 0
        
        # ES: Estadísticas de GC | EN: GC stats | JA: GC統計
        gc_counts = gc.get_count()
        
        # ES: ★ SEGURIDAD: Evitar gc.get_objects() durante inicialización
        # EN: ★ SAFETY: Avoid gc.get_objects() during initialization
        # JA: ★ 安全：初期化中は gc.get_objects() を避ける
        # ES: Solo contar objetos después de algunas muestras
        # EN: Only count objects after a few samples
        # JA: 数回サンプル後にのみオブジェクト数をカウント
        num_objects = 0
        if len(self.memory_history) >= 3 and hasattr(gc, 'get_objects'):
            try:
                # ES: Solo contar, no analizar (más rápido y seguro)
                # EN: Only count; do not analyze (faster and safer)
                # JA: カウントのみ（解析しない、より高速で安全）
                num_objects = len(gc.get_objects())
            except:
                num_objects = 0
        
        gc_stats = {
            'gen0': gc_counts[0],
            'gen1': gc_counts[1],
            'gen2': gc_counts[2],
            'objects': num_objects,
        }
        
        # Calcular frecuencia de GC
        current_time = time.time()
        if self.last_gc_time > 0:
            time_since_gc = current_time - self.last_gc_time
            if gc_counts[2] > self.stats['gc_collections']:
                self.gc_frequency = 1.0 / time_since_gc if time_since_gc > 0 else 0
                self.stats['gc_collections'] = gc_counts[2]
                self.last_gc_time = current_time
        else:
            self.last_gc_time = current_time
        
        # Detectar objetos grandes y tipos
        large_objs, type_counts = self._analyze_objects()
        
        # Calcular fragmentación proxy
        fragmentation = self._calculate_fragmentation_proxy(
            self.stats['current_memory'],
            gc_stats['objects'],
            gc_stats['gen2']
        )
        
        # ES: Guardar en historial
        # EN: Save to history
        # JP: 履歴に保存
        self.memory_history.append({
            'timestamp': timestamp,
            'memory_mb': self.stats['current_memory'],
            'memory_percent': self._get_memory_percent() if PSUTIL_AVAILABLE else 0,
        })
        
        self.fragmentation_history.append({
            'timestamp': timestamp,
            'fragmentation_score': fragmentation['score'],
            'fragmentation_ratio': fragmentation['ratio'],
            'gc_frequency': self.gc_frequency,
        })
        
        self.object_counts.append({
            'timestamp': timestamp,
            'total_objects': gc_stats['objects'],
            'large_objects': len(large_objs),
            'type_counts': dict(type_counts),
        })
        
        self.gc_stats_history.append({
            'timestamp': timestamp,
            **gc_stats
        })
        
        # ES: Guardar en log si está configurado
        # EN: Save to log if configured
        # JP: 設定されていればログに保存
        if self.log_file:
            self._write_log_entry(timestamp, self.stats['current_memory'], fragmentation, gc_stats, type_counts)
    
    def _analyze_objects(self) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Analiza objetos en memoria para detectar fragmentación
        
        Returns
        -------
        Tuple[List[Dict], Dict[str, int]]
            Lista de objetos grandes y conteo por tipo
        """
        large_objs = []
        type_counts = defaultdict(int)
        
        try:
            # Solo analizar si gc.get_objects está disponible (puede ser costoso)
            if not hasattr(gc, 'get_objects'):
                return large_objs, type_counts
            
            # ES: ★ SEGURIDAD: Evitar análisis durante inicialización
            # EN: ★ SAFETY: Avoid analysis during initialization
            # JP: ★ 安全: 初期化中は解析を避ける
            # Si el proceso acaba de empezar, esperar antes de analizar objetos
            if len(self.memory_history) < 3:  # Esperar al menos 3 muestras (15 segundos con intervalo 5s)
                return large_objs, type_counts
            
            objects = gc.get_objects()
            threshold_mb = 10  # Objetos > 10MB se consideran grandes
            
            # ★ OPTIMIZACIÓN: Limitar número de objetos analizados para evitar bloqueos
            # Analizar solo una muestra si hay muchos objetos
            max_objects_to_analyze = 50000
            if len(objects) > max_objects_to_analyze:
                # Muestrear objetos en lugar de analizar todos
                import random
                objects = random.sample(objects, max_objects_to_analyze)
            
            for obj in objects:
                obj_type = type(obj).__name__
                type_counts[obj_type] += 1
                
                # Calcular tamaño aproximado
                try:
                    size = sys.getsizeof(obj)
                    
                    # Para objetos complejos, estimar tamaño
                    if isinstance(obj, (list, tuple)):
                        size += sum(sys.getsizeof(item) for item in obj[:100])  # Muestra
                    elif isinstance(obj, dict):
                        size += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in list(obj.items())[:100])
                    elif NUMPY_AVAILABLE and isinstance(obj, np.ndarray):
                        size = obj.nbytes
                    elif PANDAS_AVAILABLE and isinstance(obj, pd.DataFrame):
                        size = obj.memory_usage(deep=True).sum()
                    
                    size_mb = size / 1024 / 1024
                    
                    if size_mb > threshold_mb:
                        large_objs.append({
                            'type': obj_type,
                            'size_mb': size_mb,
                            'id': id(obj),
                        })
                except:
                    pass
            
            # Ordenar por tamaño
            large_objs.sort(key=lambda x: x['size_mb'], reverse=True)
            large_objs = large_objs[:20]  # Top 20 objetos más grandes
            
        except Exception as e:
            # ES: El análisis de objetos puede fallar; continuar sin él
            # EN: Object analysis may fail; continue without it
            # JP: オブジェクト解析が失敗する可能性があるため、失敗しても続行する
            pass
        
        return large_objs, type_counts
    
    def _calculate_fragmentation_proxy(self, memory_mb: float, num_objects: int, gc_gen2: int) -> Dict:
        """
        Calcula métricas proxy de fragmentación
        
        La fragmentación real no es medible directamente, pero podemos inferirla:
        1. Muchos objetos pequeños pero memoria alta → fragmentación
        2. GC frecuente pero poca memoria liberada → fragmentación
        3. Memoria creciendo pero objetos pequeños → fragmentación
        
        Parameters
        ----------
        memory_mb : float
            Memoria total en MB
        num_objects : int
            Número de objetos en memoria
        gc_gen2 : int
            Número de colecciones GC generación 2
        
        Returns
        -------
        Dict
            Métricas de fragmentación proxy
        """
        # Ratio memoria/objetos (mayor = más fragmentación potencial)
        if num_objects > 0:
            memory_per_object = memory_mb / num_objects
        else:
            memory_per_object = 0
        
        # Score de fragmentación (0-100)
        # Basado en múltiples factores:
        fragmentation_score = 0.0
        
        # Factor 1: Muchos objetos pequeños (fragmentación típica)
        if num_objects > 10000 and memory_per_object < 0.1:
            fragmentation_score += 30
        
        # Factor 2: GC frecuente pero memoria alta (no se libera bien)
        if self.gc_frequency > 0.1 and memory_mb > 1000:  # GC > 1 cada 10s y memoria > 1GB
            fragmentation_score += 25
        
        # Factor 3: Memoria creciendo pero objetos pequeños
        if len(self.memory_history) > 10:
            recent_memory = [m['memory_mb'] for m in list(self.memory_history)[-10:]]
            if recent_memory[-1] > recent_memory[0] * 1.2:  # Creció 20%
                if memory_per_object < 0.2:
                    fragmentation_score += 25
        
        # Factor 4: GC gen2 frecuente (objetos de larga duración fragmentando)
        if len(self.gc_stats_history) > 5:
            recent_gc2 = [g['gen2'] for g in list(self.gc_stats_history)[-5:]]
            if recent_gc2[-1] - recent_gc2[0] > 3:  # Muchas colecciones gen2
                fragmentation_score += 20
        
        fragmentation_score = min(100.0, fragmentation_score)
        
        # Ratio de fragmentación (0-1)
        fragmentation_ratio = fragmentation_score / 100.0
        
        return {
            'score': fragmentation_score,
            'ratio': fragmentation_ratio,
            'memory_per_object': memory_per_object,
            'num_objects': num_objects,
            'gc_frequency': self.gc_frequency,
        }
    
    def _get_memory_percent(self) -> float:
        """Obtiene porcentaje de memoria usado del sistema"""
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            process = psutil.Process(self.pid)
            return process.memory_percent()
        except:
            return 0.0
    
    def _write_log_entry(self, timestamp: float, memory_mb: float, fragmentation: Dict, 
                        gc_stats: Dict, type_counts: Dict):
        """Escribe entrada en log"""
        try:
            entry = {
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                'memory_mb': memory_mb,
                'fragmentation': fragmentation,
                'gc_stats': gc_stats,
                'top_object_types': dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            }
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"⚠️ Error escribiendo log: {e}")
    
    def get_current_stats(self) -> Dict:
        """ES: Obtiene estadísticas actuales
        EN: Get current statistics
        JA: 現在の統計を取得"""
        fragmentation = {}
        if len(self.fragmentation_history) > 0:
            fragmentation = self.fragmentation_history[-1]
        
        object_info = {}
        if len(self.object_counts) > 0:
            object_info = self.object_counts[-1]
        
        gc_info = {}
        if len(self.gc_stats_history) > 0:
            gc_info = self.gc_stats_history[-1]
        
        return {
            'memory_mb': self.stats['current_memory'],
            'peak_memory_mb': self.stats['peak_memory'],
            'fragmentation': fragmentation,
            'objects': object_info,
            'gc': gc_info,
            'stats': self.stats.copy(),
        }
    
    def get_fragmentation_analysis(self) -> Dict:
        """
        Analiza qué está causando fragmentación
        
        Returns
        -------
        Dict
            Análisis de causas de fragmentación
        """
        if len(self.memory_history) < 10:
            return {'error': 'Datos insuficientes'}
        
        # Analizar tendencias
        recent_memory = [m['memory_mb'] for m in list(self.memory_history)[-20:]]
        recent_frag = [f['fragmentation_score'] for f in list(self.fragmentation_history)[-20:]]
        
        memory_trend = 'stable'
        if len(recent_memory) > 5:
            if recent_memory[-1] > recent_memory[0] * 1.3:
                memory_trend = 'growing'
            elif recent_memory[-1] < recent_memory[0] * 0.7:
                memory_trend = 'decreasing'
        
        frag_trend = 'stable'
        if len(recent_frag) > 5:
            avg_frag = sum(recent_frag) / len(recent_frag)
            if avg_frag > 50:
                frag_trend = 'high'
            elif avg_frag > 30:
                frag_trend = 'moderate'
            else:
                frag_trend = 'low'
        
        # Identificar tipos de objetos más comunes
        top_types = {}
        if len(self.object_counts) > 0:
            all_types = defaultdict(int)
            for oc in self.object_counts:
                for obj_type, count in oc.get('type_counts', {}).items():
                    all_types[obj_type] += count
            
            top_types = dict(sorted(all_types.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Causas probables de fragmentación
        causes = []
        
        if memory_trend == 'growing' and frag_trend in ['high', 'moderate']:
            causes.append({
                'type': 'memory_growth_with_fragmentation',
                'severity': 'high',
                'description': 'Memoria creciendo mientras aumenta fragmentación',
                'recommendation': 'Revisar objetos grandes que se crean frecuentemente'
            })
        
        if self.gc_frequency > 0.2:
            causes.append({
                'type': 'frequent_gc',
                'severity': 'moderate',
                'description': f'GC muy frecuente ({self.gc_frequency:.2f} veces/segundo)',
                'recommendation': 'Muchos objetos pequeños se crean/destruyen frecuentemente'
            })
        
        if 'dict' in top_types and top_types['dict'] > 10000:
            causes.append({
                'type': 'many_dicts',
                'severity': 'moderate',
                'description': f'Muchos diccionarios ({top_types["dict"]})',
                'recommendation': 'Los diccionarios pueden fragmentar el heap'
            })
        
        if 'list' in top_types and top_types['list'] > 20000:
            causes.append({
                'type': 'many_lists',
                'severity': 'moderate',
                'description': f'Muchas listas ({top_types["list"]})',
                'recommendation': 'Las listas pueden fragmentar el heap'
            })
        
        return {
            'memory_trend': memory_trend,
            'fragmentation_trend': frag_trend,
            'current_fragmentation': recent_frag[-1] if recent_frag else 0,
            'top_object_types': top_types,
            'causes': causes,
            'gc_frequency': self.gc_frequency,
        }
    
    def export_data(self, filepath: str):
        """Exporta todos los datos a JSON"""
        data = {
            'stats': self.stats,
            'memory_history': list(self.memory_history),
            'fragmentation_history': list(self.fragmentation_history),
            'object_counts': list(self.object_counts),
            'gc_stats_history': list(self.gc_stats_history),
            'analysis': self.get_fragmentation_analysis(),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ データをエクスポートしました: {filepath}")

