[Setup]
#ifndef BuildMode
  #define BuildMode "onedir"
#endif
#ifndef IncludeVCRedist
  #define IncludeVCRedist 1
#endif
#ifndef IncludeWizardImages
  #define IncludeWizardImages 1
#endif
#ifndef IncludeSetupIcon
  #define IncludeSetupIcon 1
#endif

AppName=0.00sec
AppPublisher=XebecTechnology
AppVersion=1.0.0
DefaultDirName={autopf}\XebecTechnology\0.00sec
DefaultGroupName=XebecTechnology\0.00sec
; Dejar el instalador generado dentro de la carpeta `installer\`
OutputDir={#SourcePath}
; Nombre único por build para evitar bloqueos/archivos truncados si se ejecuta mientras compila
OutputBaseFilename=0_00sec_Setup_x64_{#GetDateTimeString('yyyyMMdd_HHmmss', '', '')}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
DisableProgramGroupPage=yes
WizardStyle=modern
ShowLanguageDialog=yes
; Icono del instalador (barra de título / UAC)
#if IncludeSetupIcon
SetupIconFile=setup_icon.ico
#endif
; Logo(s) en el wizard (requiere BMP; se generan por script en `installer\wizard_*.bmp`)
#if IncludeWizardImages
WizardSmallImageFile=wizard_small.bmp
WizardImageFile=wizard_large.bmp
#endif

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[CustomMessages]
spanish.ShortcutsGroup=Accesos directos:
spanish.DesktopShortcutTask=Crear acceso directo en el Escritorio
spanish.RunApp=Ejecutar 0.00sec

japanese.ShortcutsGroup=ショートカット:
japanese.DesktopShortcutTask=デスクトップにショートカットを作成する
japanese.RunApp=0.00sec を実行

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopShortcutTask}"; GroupDescription: "{cm:ShortcutsGroup}"; Flags: unchecked

[Dirs]
; Datos compartidos (se conservan al desinstalar)
Name: "{commonappdata}\XebecTechnology\0.00sec"; Permissions: users-modify; Flags: uninsneveruninstall
Name: "{commonappdata}\XebecTechnology\0.00sec\data"; Permissions: users-modify; Flags: uninsneveruninstall
Name: "{commonappdata}\XebecTechnology\0.00sec\backups"; Permissions: users-modify; Flags: uninsneveruninstall

[Files]
; Runtime VC++ (por si el PC destino no lo tiene; evita errores al iniciar el intérprete embebido)
#if IncludeVCRedist
Source: "vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: VCRedistNeeded
#endif

; BBDD vacías (se copian SOLO si no existen; se conservan al desinstalar)
Source: "db_templates\\results.db"; DestDir: "{commonappdata}\\XebecTechnology\\0.00sec\\data"; Flags: onlyifdoesntexist uninsneveruninstall ignoreversion
Source: "db_templates\\yosoku_predictions_lineal.db"; DestDir: "{commonappdata}\\XebecTechnology\\0.00sec\\data"; Flags: onlyifdoesntexist uninsneveruninstall ignoreversion
Source: "db_templates\\yosoku_predictions_no_lineal.db"; DestDir: "{commonappdata}\\XebecTechnology\\0.00sec\\data"; Flags: onlyifdoesntexist uninsneveruninstall ignoreversion

; Copiar build de PyInstaller (onedir vs onefile)
#if BuildMode == "onefile"
Source: "..\dist\0_00sec.exe"; DestDir: "{app}"; Flags: ignoreversion
#else
Source: "..\dist\0_00sec\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
#endif

[Icons]
Name: "{group}\0.00sec"; Filename: "{app}\0_00sec.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\0.00sec"; Filename: "{app}\0_00sec.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
#if IncludeVCRedist
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; Flags: waituntilterminated; Check: VCRedistNeeded
#endif
Filename: "{app}\0_00sec.exe"; WorkingDir: "{app}"; Description: "{cm:RunApp}"; Flags: nowait postinstall skipifsilent

[Code]
function VCRedistInstalled: Boolean;
var
  installed: Cardinal;
begin
  Result :=
    RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', installed)
    and (installed = 1);
end;

function VCRedistNeeded: Boolean;
begin
  Result := not VCRedistInstalled;
end;


