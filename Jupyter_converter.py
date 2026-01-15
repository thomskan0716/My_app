#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jupyter Converter GUI — Selecciona archivo o carpeta y convierte .ipynb -> .py
Guarda el .py en la misma carpeta del .ipynb.
"""

from pathlib import Path
import nbformat
from nbconvert import PythonExporter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

def find_ipynb_targets(root: Path, recursive: bool) -> list[Path]:
    """Encuentra .ipynb bajo root (evita .ipynb_checkpoints)."""
    if root.is_file():
        return [root] if root.suffix.lower() == ".ipynb" else []
    pattern = "**/*.ipynb" if recursive else "*.ipynb"
    return [p for p in root.glob(pattern) if ".ipynb_checkpoints" not in p.parts]

def convert_notebook(ipynb_path: Path, code_only: bool, overwrite: bool, logger=None) -> Path | None:
    """Convierte un .ipynb a .py (misma carpeta)."""
    try:
        if not ipynb_path.exists() or ipynb_path.suffix.lower() != ".ipynb":
            if logger: logger(f"[SKIP] No es .ipynb válido: {ipynb_path}")
            return None

        nb = nbformat.read(ipynb_path, as_version=4)
        if code_only:
            nb.cells = [c for c in nb.cells if c.cell_type == "code"]

        exporter = PythonExporter()
        body, _ = exporter.from_notebook_node(nb)

        out_path = ipynb_path.with_suffix(".py")
        if out_path.exists() and not overwrite:
            if logger: logger(f"[SKIP] Ya existe (activa 'Sobrescribir'): {out_path}")
            return None

        out_path.write_text(body, encoding="utf-8")
        if logger: logger(f"[OK] {ipynb_path}  ->  {out_path}")
        return out_path

    except Exception as e:
        if logger: logger(f"[ERROR] {ipynb_path}: {e}")
        return None

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Jupyter Converter (.ipynb → .py)")
        self.geometry("720x420")

        self.file_var = tk.StringVar()
        self.dir_var = tk.StringVar()
        self.recursive_var = tk.BooleanVar(value=True)
        self.code_only_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)

        # Selección de archivo
        ttk.Label(frm, text="Archivo .ipynb:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.file_var, width=70).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Elegir archivo…", command=self.browse_file).grid(row=0, column=2)

        # Selección de carpeta
        ttk.Label(frm, text="Carpeta notebooks:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.dir_var, width=70).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Elegir carpeta…", command=self.browse_folder).grid(row=1, column=2)

        # Opciones
        opt = ttk.Frame(self)
        opt.pack(fill="x", **pad)
        ttk.Checkbutton(opt, text="Recursivo (subcarpetas)", variable=self.recursive_var).pack(side="left")
        ttk.Checkbutton(opt, text="Solo código (sin Markdown)", variable=self.code_only_var).pack(side="left", padx=12)
        ttk.Checkbutton(opt, text="Sobrescribir si existe", variable=self.overwrite_var).pack(side="left", padx=12)

        # Botón convertir
        actions = ttk.Frame(self)
        actions.pack(fill="x", **pad)
        ttk.Button(actions, text="Convertir", command=self.run_convert).pack(side="left")
        ttk.Button(actions, text="Salir", command=self.destroy).pack(side="right")

        # Log
        self.log_box = ScrolledText(self, height=14, wrap="none")
        self.log_box.pack(fill="both", expand=True, padx=8, pady=6)
        self.log("Listo. Elige un archivo .ipynb o una carpeta y pulsa Convertir.")

        # Estirar columnas
        frm.grid_columnconfigure(1, weight=1)

    def log(self, text: str):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.update_idletasks()

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Selecciona un notebook",
            filetypes=[("Jupyter Notebook", "*.ipynb"), ("Todos los archivos", "*.*")]
        )
        if path:
            self.file_var.set(path)

    def browse_folder(self):
        path = filedialog.askdirectory(title="Selecciona carpeta con notebooks")
        if path:
            self.dir_var.set(path)

    def run_convert(self):
        file_path = self.file_var.get().strip()
        dir_path = self.dir_var.get().strip()

        targets: list[Path] = []
        if file_path:
            p = Path(file_path)
            if p.suffix.lower() != ".ipynb":
                messagebox.showwarning("Archivo no válido", "Selecciona un archivo .ipynb.")
                return
            targets = [p]
        elif dir_path:
            targets = find_ipynb_targets(Path(dir_path), self.recursive_var.get())
        else:
            messagebox.showinfo("Falta selección", "Elige un archivo .ipynb o una carpeta.")
            return

        if not targets:
            self.log("[INFO] No se encontraron .ipynb para convertir.")
            return

        self.log(f"[INFO] Encontrados {len(targets)} notebook(s). Iniciando conversión…")
        ok, skip, err = 0, 0, 0
        for nb in targets:
            res = convert_notebook(
                ipynb_path=nb,
                code_only=self.code_only_var.get(),
                overwrite=self.overwrite_var.get(),
                logger=self.log
            )
            if res is None:
                # Heurística para contar skips/errores por el log:
                # (no perfecto, pero informativo)
                # Aquí no diferenciamos bien sin estados; está bien mantener simple.
                pass
            else:
                ok += 1

        self.log(f"[RESUMEN] Convertidos OK: {ok}. Revisar líneas [SKIP]/[ERROR] en el log.")
        messagebox.showinfo("Terminado", f"Conversión finalizada.\nCorrectos: {ok}\nRevisa el log para SKIP/ERROR.")

if __name__ == "__main__":
    App().mainloop()
