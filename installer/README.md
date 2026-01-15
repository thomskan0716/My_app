## Instalador (Inno Setup)

Este proyecto genera un instalador profesional `Setup.exe` para `0.00sec` que:

- Instala la app en `C:\Program Files\XebecTechnology\0.00sec`
- Crea datos compartidos en `C:\ProgramData\XebecTechnology\0.00sec\{data,backups}`
- Da permisos de escritura a `Users` en `ProgramData` (necesario para SQLite y backups)
- Pregunta si crear acceso directo en Escritorio
- **Conserva** la BBDD y backups al desinstalar

### Requisitos

- Inno Setup instalado (incluye `ISCC.exe`)

### Pasos

#### Opción recomendada (un solo comando)

Ejecuta desde la carpeta raíz del proyecto:

`build_installer.ps1`

Por defecto:
- Construye `PyInstaller` en modo **onedir**
- Genera los logos del wizard del instalador desde `Square44x44Logo.scale-400.png`
- Descarga `vc_redist.x64.exe` (VC++ 2015-2022 x64) para evitar fallos de arranque en PCs “limpios”
- Compila `installer/setup.iss`

Parámetros útiles:
- `-Mode onedir|onefile` (onefile hace el instalador más simple; onedir suele arrancar más rápido)
- `-NoInstall` (no hace `pip install/upgrade` de herramientas)
- `-InstallerOnly` (solo compila el instalador usando el `dist/` existente; útil para iterar rápido)
- `-SkipClean` (no borra `build/` y `dist/` antes del build)
- `-SkipVCRedistDownload` (compila sin incluir VC++ redist)
- `-SkipWizardImages` (compila sin logos en el wizard)

#### Manual (si quieres separar pasos)

1) Genera el build del EXE (PyInstaller onedir):

`build_exe_onedir.ps1`

2) Abre `installer/setup.iss` en Inno Setup y compílalo, o ejecuta:

`ISCC.exe installer\setup.iss /DBuildMode=onedir`

El instalador se generará como `installer\0_00sec_Setup_x64.exe` (según `OutputBaseFilename` y `OutputDir`).



