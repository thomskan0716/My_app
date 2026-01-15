"""
Visualizador de memoria y fragmentación del heap
Genera gráficos a partir de los datos exportados por MemoryMonitor
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def load_memory_data(json_path: str) -> dict:
    """Carga datos de memoria desde JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_memory_timeline(data: dict, output_path: str):
    """Grafica memoria y fragmentación en el tiempo"""
    memory_history = data.get('memory_history', [])
    fragmentation_history = data.get('fragmentation_history', [])
    
    if not memory_history:
        print("⚠️ No hay datos de memoria para graficar")
        return
    
    # Extraer datos
    timestamps = [m['timestamp'] for m in memory_history]
    memory_mb = [m['memory_mb'] for m in memory_history]
    frag_scores = [f.get('fragmentation_score', 0) for f in fragmentation_history]
    
    # Normalizar timestamps (empezar desde 0)
    start_time = timestamps[0] if timestamps else 0
    time_elapsed = [(t - start_time) / 60 for t in timestamps]  # Convertir a minutos
    
    # Crear figura con subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Monitoreo de Memoria y Fragmentación del Heap', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Memoria RSS
    ax1 = axes[0]
    ax1.plot(time_elapsed, memory_mb, 'b-', linewidth=2, label='Memoria RSS')
    ax1.fill_between(time_elapsed, memory_mb, alpha=0.3)
    ax1.set_xlabel('Tiempo (minutos)')
    ax1.set_ylabel('Memoria (MB)')
    ax1.set_title('Memoria del Proceso')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Añadir línea de pico
    peak_memory = max(memory_mb)
    peak_idx = memory_mb.index(peak_memory)
    ax1.axhline(y=peak_memory, color='r', linestyle='--', alpha=0.5, label=f'Pico: {peak_memory:.1f}MB')
    ax1.annotate(f'Pico: {peak_memory:.1f}MB', 
                xy=(time_elapsed[peak_idx], peak_memory),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->'))
    
    # Gráfico 2: Fragmentación
    ax2 = axes[1]
    if len(frag_scores) == len(time_elapsed):
        ax2.plot(time_elapsed, frag_scores, 'r-', linewidth=2, label='Score de Fragmentación')
        ax2.fill_between(time_elapsed, frag_scores, alpha=0.3, color='red')
        
        # Zonas de riesgo
        ax2.axhspan(0, 30, alpha=0.1, color='green', label='Baja')
        ax2.axhspan(30, 50, alpha=0.1, color='yellow', label='Moderada')
        ax2.axhspan(50, 100, alpha=0.1, color='red', label='Alta')
        
        ax2.set_xlabel('Tiempo (minutos)')
        ax2.set_ylabel('Fragmentación (%)')
        ax2.set_title('Fragmentación del Heap (Proxy)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Gráfico 3: Número de objetos
    object_counts = data.get('object_counts', [])
    if object_counts:
        num_objects = [oc.get('total_objects', 0) for oc in object_counts]
        if len(num_objects) == len(time_elapsed):
            ax3 = axes[2]
            ax3.plot(time_elapsed, num_objects, 'g-', linewidth=2, label='Total de Objetos')
            ax3.set_xlabel('Tiempo (minutos)')
            ax3.set_ylabel('Número de Objetos')
            ax3.set_title('Objetos en Memoria')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {output_path}")
    plt.close()


def plot_fragmentation_causes(data: dict, output_path: str):
    """Grafica análisis de causas de fragmentación"""
    analysis = data.get('analysis', {})
    
    if 'causes' not in analysis or not analysis['causes']:
        print("⚠️ No hay causas de fragmentación identificadas")
        return
    
    causes = analysis['causes']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Preparar datos
    cause_types = [c['type'] for c in causes]
    severities = [c['severity'] for c in causes]
    
    # Colores según severidad
    colors = {'high': 'red', 'moderate': 'orange', 'low': 'yellow'}
    bar_colors = [colors.get(s, 'gray') for s in severities]
    
    # Crear gráfico de barras
    y_pos = np.arange(len(cause_types))
    bars = ax.barh(y_pos, [1] * len(cause_types), color=bar_colors, alpha=0.7)
    
    # Añadir descripciones
    descriptions = [c['description'] for c in causes]
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{t}\n{d}" for t, d in zip(cause_types, descriptions)], fontsize=9)
    ax.set_xlabel('Severidad')
    ax.set_title('Causas Identificadas de Fragmentación del Heap', fontsize=14, fontweight='bold')
    
    # Leyenda de severidad
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Alta'),
        Patch(facecolor='orange', alpha=0.7, label='Moderada'),
        Patch(facecolor='yellow', alpha=0.7, label='Baja')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico de causas guardado: {output_path}")
    plt.close()


def plot_object_types(data: dict, output_path: str):
    """Grafica distribución de tipos de objetos"""
    object_counts = data.get('object_counts', [])
    
    if not object_counts:
        print("⚠️ No hay datos de tipos de objetos")
        return
    
    # Acumular tipos de objetos
    type_totals = {}
    for oc in object_counts:
        type_counts = oc.get('type_counts', {})
        for obj_type, count in type_counts.items():
            type_totals[obj_type] = type_totals.get(obj_type, 0) + count
    
    if not type_totals:
        print("⚠️ No hay tipos de objetos registrados")
        return
    
    # Top 15 tipos
    sorted_types = sorted(type_totals.items(), key=lambda x: x[1], reverse=True)[:15]
    types, counts = zip(*sorted_types) if sorted_types else ([], [])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(types))
    bars = ax.barh(y_pos, counts, color='steelblue', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types)
    ax.set_xlabel('Conteo Total')
    ax.set_title('Tipos de Objetos Más Comunes en Memoria', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores en las barras
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f' {count:,}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico de tipos guardado: {output_path}")
    plt.close()


def generate_report(json_path: str, output_dir: str = None):
    """
    Genera reporte completo de memoria y fragmentación
    
    Parameters
    ----------
    json_path : str
        Ruta al archivo JSON con datos de memoria
    output_dir : str, optional
        Directorio de salida (por defecto, mismo que JSON)
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"❌ Archivo no encontrado: {json_path}")
        return
    
    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generando reporte de memoria desde: {json_path}")
    
    # Cargar datos
    data = load_memory_data(str(json_path))
    
    # Generar gráficos
    plot_memory_timeline(data, str(output_dir / 'memory_timeline.png'))
    plot_fragmentation_causes(data, str(output_dir / 'fragmentation_causes.png'))
    plot_object_types(data, str(output_dir / 'object_types.png'))
    
    # Generar reporte de texto
    report_path = output_dir / 'memory_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("REPORTE DE MEMORIA Y FRAGMENTACIÓN\n")
        f.write("="*60 + "\n\n")
        
        stats = data.get('stats', {})
        f.write(f"Memoria Pico: {stats.get('peak_memory', 0):.1f} MB\n")
        f.write(f"Memoria Actual: {stats.get('current_memory', 0):.1f} MB\n")
        f.write(f"Total de Asignaciones: {stats.get('total_allocations', 0)}\n")
        f.write(f"Total de Desasignaciones: {stats.get('total_deallocations', 0)}\n")
        f.write(f"Colecciones GC: {stats.get('gc_collections', 0)}\n")
        f.write(f"Eventos de Fragmentación: {stats.get('fragmentation_events', 0)}\n\n")
        
        analysis = data.get('analysis', {})
        f.write("ANÁLISIS DE FRAGMENTACIÓN\n")
        f.write("-"*60 + "\n")
        f.write(f"Tendencia de Memoria: {analysis.get('memory_trend', 'N/A')}\n")
        f.write(f"Tendencia de Fragmentación: {analysis.get('fragmentation_trend', 'N/A')}\n")
        f.write(f"Fragmentación Actual: {analysis.get('current_fragmentation', 0):.1f}%\n")
        f.write(f"Frecuencia GC: {analysis.get('gc_frequency', 0):.2f} veces/segundo\n\n")
        
        if 'causes' in analysis and analysis['causes']:
            f.write("CAUSAS IDENTIFICADAS:\n")
            f.write("-"*60 + "\n")
            for i, cause in enumerate(analysis['causes'], 1):
                f.write(f"{i}. {cause['description']}\n")
                f.write(f"   Severidad: {cause['severity']}\n")
                f.write(f"   Recomendación: {cause['recommendation']}\n\n")
        
        if 'top_object_types' in analysis:
            f.write("TIPOS DE OBJETOS MÁS COMUNES:\n")
            f.write("-"*60 + "\n")
            for obj_type, count in list(analysis['top_object_types'].items())[:10]:
                f.write(f"  {obj_type}: {count:,}\n")
    
    print(f"✅ Reporte de texto guardado: {report_path}")
    print(f"\n✅ Reporte completo generado en: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python memory_visualizer.py <ruta_al_json> [directorio_salida]")
        print("\nEjemplo:")
        print("  python memory_visualizer.py memory_analysis.json")
        print("  python memory_visualizer.py memory_analysis.json ./reportes")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_report(json_path, output_dir)








