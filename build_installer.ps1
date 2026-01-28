param(
  [ValidateSet("onedir","onefile")]
  [string]$Mode = "onedir",
  [switch]$NoInstall,
  [switch]$SkipVCRedistDownload,
  [switch]$SkipWizardImages,
  [switch]$SkipSetupIcon,
  [switch]$InstallerOnly,
  [switch]$SkipClean
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$python = Join-Path $here "Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "venv の Python が見つかりません: $python（このスクリプトはプロジェクトの .venv フォルダーから実行してください）"
}

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path $Path)) { New-Item -ItemType Directory -Path $Path | Out-Null }
}

function Find-ISCC {
  $candidates = @(
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
  )
  foreach ($p in $candidates) {
    if ($p -and (Test-Path $p)) { return $p }
  }
  return $null
}

if (-not $NoInstall) {
  Write-Host "ビルド依存関係をインストール／更新中（PyInstaller + Pillow）..."
  & $python -m pip install --upgrade pyinstaller pillow | Out-Host
}

# 1) Generar BMPs del wizard desde Square44x44Logo.scale-400.png (opcional)
$installerDir = Join-Path $here "installer"
Ensure-Dir $installerDir

$srcLogo = Join-Path $here "Square44x44Logo.scale-400.png"
$wizardSmall = Join-Path $installerDir "wizard_small.bmp"
$wizardLarge = Join-Path $installerDir "wizard_large.bmp"
$setupIcon = Join-Path $installerDir "setup_icon.ico"

# 0) Generar BBDD vacías (templates) para el instalador
$dbTplDir = Join-Path $installerDir "db_templates"
Ensure-Dir $dbTplDir

Write-Host "インストーラー用の空DBを生成中（installer\\db_templates\\）..."
$pyTmp = Join-Path $env:TEMP ("gen_db_templates_{0}.py" -f ([Guid]::NewGuid().ToString("N")))
$pyCode = @'
import os, sqlite3

tpl_dir = os.path.abspath(os.path.join("installer", "db_templates"))
os.makedirs(tpl_dir, exist_ok=True)

def init_results_db(path: str):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass

    cur.execute("""
        CREATE TABLE IF NOT EXISTS main_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            "\u5b9f\u9a13\u65e5" TEXT,
            "\u30d0\u30ea\u9664\u53bb" INTEGER,
            "\u4e0a\u9762\u30c0\u30ec\u91cf" REAL,
            "\u5074\u9762\u30c0\u30ec\u91cf" REAL,
            "\u6469\u8017\u91cf" REAL,
            "\u9762\u7c97\u5ea6\u524d" REAL,
            "\u9762\u7c97\u5ea6\u5f8c" REAL,
            A13 INTEGER,
            A11 INTEGER,
            A21 INTEGER,
            A32 INTEGER,
            "\u76f4\u5f84" REAL,
            "\u6750\u6599" TEXT,
            "\u7dda\u6750\u9577" INTEGER,
            "\u56de\u8ee2\u901f\u5ea6" INTEGER,
            "\u9001\u308a\u901f\u5ea6" INTEGER,
            "UP\u30ab\u30c3\u30c8" INTEGER,
            "\u5207\u8fbc\u91cf" REAL,
            "\u7a81\u51fa\u91cf" INTEGER,
            "\u8f09\u305b\u7387" REAL,
            "\u30d1\u30b9\u6570" INTEGER,
            "\u52a0\u5de5\u6642\u9593" REAL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS Results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            "\u5b9f\u9a13\u65e5" INTEGER,
            "\u30d0\u30ea\u9664\u53bb" INTEGER,
            "\u4e0a\u9762\u30c0\u30ec\u91cf" REAL,
            "\u5074\u9762\u30c0\u30ec\u91cf" REAL,
            "\u9762\u7c97\u5ea6\u524d" TEXT,
            "\u9762\u7c97\u5ea6\u5f8c" TEXT,
            "\u6469\u8017\u91cf" REAL,
            "\u6750\u6599" TEXT,
            A13 INTEGER,
            A11 INTEGER,
            A21 INTEGER,
            A32 INTEGER,
            "\u76f4\u5f84" REAL,
            "\u56de\u8ee2\u901f\u5ea6" INTEGER,
            "\u9001\u308a\u901f\u5ea6" INTEGER,
            "UP\u30ab\u30c3\u30c8" INTEGER,
            "\u5207\u8fbc\u91cf" REAL,
            "\u7a81\u51fa\u91cf" INTEGER,
            "\u8f09\u305b\u7387" REAL,
            "\u30d1\u30b9\u6570" INTEGER,
            "\u7dda\u6750\u9577" INTEGER,
            "\u52a0\u5de5\u6642\u9593" REAL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS TemporaryResults (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            "\u5b9f\u9a13\u65e5" INTEGER,
            "\u30d0\u30ea\u9664\u53bb" INTEGER,
            "\u4e0a\u9762\u30c0\u30ec\u91cf" REAL,
            "\u5074\u9762\u30c0\u30ec\u91cf" REAL,
            "\u9762\u7c97\u5ea6\u524d" TEXT,
            "\u9762\u7c97\u5ea6\u5f8c" TEXT,
            "\u6469\u8017\u91cf" REAL,
            "\u6750\u6599" TEXT,
            A13 INTEGER,
            A11 INTEGER,
            A21 INTEGER,
            A32 INTEGER,
            "\u76f4\u5f84" REAL,
            "\u56de\u8ee2\u901f\u5ea6" INTEGER,
            "\u9001\u308a\u901f\u5ea6" INTEGER,
            "UP\u30ab\u30c3\u30c8" INTEGER,
            "\u5207\u8fbc\u91cf" REAL,
            "\u7a81\u51fa\u91cf" INTEGER,
            "\u8f09\u305b\u7387" REAL,
            "\u30d1\u30b9\u6570" INTEGER,
            "\u7dda\u6750\u9577" INTEGER,
            "\u52a0\u5de5\u6642\u9593" REAL
        );
    """)

    conn.commit()
    conn.close()

def init_yosoku_db(path: str):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass

    cur.execute("""
        CREATE TABLE IF NOT EXISTS yosoku_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            A13 INTEGER,
            A11 INTEGER,
            A21 INTEGER,
            A32 INTEGER,
            "\u76f4\u5f84" REAL,
            "\u6750\u6599" TEXT,
            "\u7dda\u6750\u9577" REAL,
            "\u56de\u8ee2\u901f\u5ea6" REAL,
            "\u9001\u308a\u901f\u5ea6" REAL,
            "UP\u30ab\u30c3\u30c8" INTEGER,
            "\u5207\u8fbc\u91cf" REAL,
            "\u7a81\u51fa\u91cf" REAL,
            "\u8f09\u305b\u7387" REAL,
            "\u30d1\u30b9\u6570" INTEGER,
            "\u52a0\u5de5\u6642\u9593" REAL,
            "\u4e0a\u9762\u30c0\u30ec\u91cf" REAL,
            "\u5074\u9762\u30c0\u30ec\u91cf" REAL,
            "\u6469\u8017\u91cf" REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pred_label INTEGER,
            p_cal REAL,
            tau_pos REAL,
            tau_neg REAL,
            ood_flag INTEGER,
            maha_dist REAL
        );
    """)

    cur.execute("DROP INDEX IF EXISTS idx_unique_yosoku;")
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_yosoku
        ON yosoku_predictions (
            A13, A11, A21, A32, "\u76f4\u5f84", "\u6750\u6599", "\u7dda\u6750\u9577", "\u56de\u8ee2\u901f\u5ea6",
            "\u9001\u308a\u901f\u5ea6", "UP\u30ab\u30c3\u30c8", "\u5207\u8fbc\u91cf", "\u7a81\u51fa\u91cf",
            "\u8f09\u305b\u7387", "\u30d1\u30b9\u6570", "\u52a0\u5de5\u6642\u9593"
        );
    """)

    conn.commit()
    conn.close()

results_db = os.path.join(tpl_dir, "results.db")
yosoku_lin = os.path.join(tpl_dir, "yosoku_predictions_lineal.db")
yosoku_non = os.path.join(tpl_dir, "yosoku_predictions_no_lineal.db")

init_results_db(results_db)
init_yosoku_db(yosoku_lin)
init_yosoku_db(yosoku_non)

print("OK:", results_db)
print("OK:", yosoku_lin)
print("OK:", yosoku_non)
'@
Set-Content -Path $pyTmp -Value $pyCode -Encoding UTF8
try {
  & $python $pyTmp | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "Falló la generación de BBDD templates." }
} finally {
  Remove-Item -Force $pyTmp -ErrorAction SilentlyContinue
}

# 1a) Generar icono del instalador (SetupIconFile) desde el PNG (opcional)
if (-not $SkipSetupIcon) {
  if (-not (Test-Path $srcLogo)) {
    throw "No se encontró el logo fuente para el icono: $srcLogo"
  }

  Write-Host "Generando icono del instalador (installer\\setup_icon.ico)..."
  $pyTmp = Join-Path $env:TEMP ("gen_setup_icon_{0}.py" -f ([Guid]::NewGuid().ToString("N")))
  $pyCode = @'
from PIL import Image
import os

src = os.path.abspath("Square44x44Logo.scale-400.png")
out_ico = os.path.abspath(os.path.join("installer", "setup_icon.ico"))

img = Image.open(src).convert("RGBA")
img.save(out_ico, format="ICO", sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
print("OK:", out_ico)
'@
  Set-Content -Path $pyTmp -Value $pyCode -Encoding UTF8
  try {
    & $python $pyTmp | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Falló la generación de setup_icon.ico" }
  } finally {
    Remove-Item -Force $pyTmp -ErrorAction SilentlyContinue
  }
} else {
  Write-Host "SkipSetupIcon 有効: setup_icon.ico は生成しません"
}

if (-not $SkipWizardImages) {
  if (-not (Test-Path $srcLogo)) {
    throw "No se encontró el logo: $srcLogo"
  }

  Write-Host "インストーラー画像を生成中（wizard_small.bmp / wizard_large.bmp）..."
  $pyTmp = Join-Path $env:TEMP ("gen_wizard_images_{0}.py" -f ([Guid]::NewGuid().ToString("N")))
  $pyCode = @'
from PIL import Image
import os

src = os.path.abspath("Square44x44Logo.scale-400.png")
out_small = os.path.abspath(os.path.join("installer", "wizard_small.bmp"))
out_large = os.path.abspath(os.path.join("installer", "wizard_large.bmp"))

logo = Image.open(src).convert("RGBA")

def make_canvas(size, bg=(255, 255, 255, 255)):
    return Image.new("RGBA", size, bg)

def center_paste(canvas, img, y_offset=0):
    cx = (canvas.width - img.width) // 2
    cy = (canvas.height - img.height) // 2 + y_offset
    canvas.alpha_composite(img, (cx, cy))

# Small (Inno Setup recomendado: 55x55)
small = make_canvas((55, 55))
logo_s = logo.copy()
logo_s.thumbnail((45, 45), Image.LANCZOS)
center_paste(small, logo_s, 0)
small.convert("RGB").save(out_small, format="BMP")

# Large (Inno Setup recomendado: 164x314) - logo arriba, centrado
large = make_canvas((164, 314))
logo_l = logo.copy()
logo_l.thumbnail((120, 120), Image.LANCZOS)
center_paste(large, logo_l, y_offset=-70)
large.convert("RGB").save(out_large, format="BMP")

print("OK:", out_small)
print("OK:", out_large)
'@
  Set-Content -Path $pyTmp -Value $pyCode -Encoding UTF8
  try {
    & $python $pyTmp | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Falló la generación de BMPs del instalador." }
  } finally {
    Remove-Item -Force $pyTmp -ErrorAction SilentlyContinue
  }
} else {
  Write-Host "SkipWizardImages 有効: wizard_small.bmp / wizard_large.bmp は生成しません"
}

# 2) Descargar VC++ Redist (si no existe)
$vcredist = Join-Path $installerDir "vc_redist.x64.exe"
if (-not $SkipVCRedistDownload) {
  if (-not (Test-Path $vcredist)) {
    $url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    Write-Host "Descargando VC++ Redistributable (x64) -> $vcredist"
    Invoke-WebRequest -Uri $url -OutFile $vcredist
  } else {
    Write-Host "VC++ Redistributable ya existe: $vcredist"
  }
} else {
  Write-Host "SkipVCRedistDownload 有効: vc_redist.x64.exe はダウンロードしません"
}

if (-not $InstallerOnly) {
  # ES: 3) Limpiar build/dist para evitar artefactos obsoletos (opcional)
  # EN: 3) Clean build/dist to avoid stale artifacts (optional)
  # JP: 3) 古い生成物を避けるためbuild/distをクリーン（任意）
  if (-not $SkipClean) {
    foreach ($d in @("build", "dist")) {
      $p = Join-Path $here $d
      if (Test-Path $p) {
        Write-Host "$p をクリーン中..."
        Remove-Item -Recurse -Force $p
      }
    }
  } else {
    Write-Host "SkipClean 有効: build/dist は削除しません"
  }

  # 4) Build PyInstaller
  if ($Mode -eq "onefile") {
    Write-Host "Construyendo ejecutable (onefile)..."
    & (Join-Path $here "build_exe_onefile.ps1") -NoInstall | Out-Host
  } else {
    Write-Host "Construyendo ejecutable (onedir)..."
    & (Join-Path $here "build_exe_onedir.ps1") -NoInstall | Out-Host
  }
  if ($LASTEXITCODE -ne 0) { throw "PyInstaller falló (exit code $LASTEXITCODE)" }
} else {
  # ES: Validar que exista el artefacto esperado en dist/; si no, forzar build.
  # EN: Validate the expected artifact exists in dist/; otherwise, force a build.
  # JP: dist/に期待する成果物があるか確認し、無ければビルドを強制する
  $expectedOk = $false
  if ($Mode -eq "onefile") {
    $expectedOk = Test-Path (Join-Path $here "dist\0_00sec.exe")
  } else {
    $expectedOk = Test-Path (Join-Path $here "dist\0_00sec\0_00sec.exe")
  }

  if (-not $expectedOk) {
    Write-Host "InstallerOnly 有効ですが dist/ にビルドがありません。先に PyInstaller を実行します..."
    $InstallerOnly = $false

    if (-not $SkipClean) {
      foreach ($d in @("build", "dist")) {
        $p = Join-Path $here $d
        if (Test-Path $p) {
          Write-Host "$p をクリーン中..."
          Remove-Item -Recurse -Force $p
        }
      }
    } else {
      Write-Host "SkipClean 有効: build/dist は削除しません"
    }

    if ($Mode -eq "onefile") {
      Write-Host "Construyendo ejecutable (onefile)..."
      & (Join-Path $here "build_exe_onefile.ps1") -NoInstall | Out-Host
    } else {
      Write-Host "Construyendo ejecutable (onedir)..."
      & (Join-Path $here "build_exe_onedir.ps1") -NoInstall | Out-Host
    }
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller falló (exit code $LASTEXITCODE)" }
  } else {
    Write-Host "InstallerOnly 有効: PyInstaller をスキップします（既存の dist を使用）"
  }
}

# 5) Compilar instalador (Inno Setup)
$iscc = Find-ISCC
if (-not $iscc) {
  throw "No se encontró ISCC.exe (Inno Setup). Instala Inno Setup 6 y vuelve a ejecutar. Rutas probadas: '$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe' y '$env:ProgramFiles\Inno Setup 6\ISCC.exe'."
}

Write-Host "Compilando instalador con Inno Setup..."
$defs = @("/DBuildMode=$Mode")
if ($SkipVCRedistDownload) { $defs += "/DIncludeVCRedist=0" }
if ($SkipWizardImages) { $defs += "/DIncludeWizardImages=0" }
if ($SkipSetupIcon) { $defs += "/DIncludeSetupIcon=0" }

& $iscc (Join-Path $installerDir "setup.iss") @defs | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Inno Setup falló (exit code $LASTEXITCODE)" }

Write-Host ""
Write-Host "✅ インストーラーを生成しました: installer\\（ファイル: 0_00sec_Setup_x64_*.exe）"


