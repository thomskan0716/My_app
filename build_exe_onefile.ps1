param(
  [switch]$NoInstall
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$python = Join-Path $here "Scripts\\python.exe"
if (-not (Test-Path $python)) {
  throw "venv の Python が見つかりません: $python（このスクリプトはプロジェクトの .venv フォルダーから実行してください）"
}

if (-not $NoInstall) {
  Write-Host "venv に PyInstaller をインストール／更新中..."
  & $python -m pip install --upgrade pyinstaller
}

Write-Host ""
Write-Host "注意: --onefile では解凍のため起動直後に遅延が発生し、Qt がスプラッシュを表示するまで時間がかかる場合があります。"
Write-Host ""

$name = "0_00sec"
$icon = "xebec.ico"
$args = @(
  "--noconfirm",
  "--clean",
  "--onefile",
  "--windowed",
  "--name", $name,
  "--hidden-import", "0sec",
  # Matplotlib/kiwisolver: asegurar extensión compilada
  "--hidden-import", "kiwisolver._cext",
  # Evitar que PyInstaller intente empaquetar múltiples bindings de Qt (PyQt5/PyQt6)
  "--exclude-module", "PyQt5",
  "--exclude-module", "PyQt6",
  "--add-data", "SlpashScreen_General.png;.",
  "--add-data", "SplashScreen.scale-100.png;.",
  "--add-data", "xebec_logo_88.png;.",
  "--add-data", "loading.gif;."
)

if (Test-Path $icon) {
  $args += @("--icon", $icon, "--add-data", "$icon;.")
}

$optionalData = @(
  @{ src = "manifest.json"; dest = "." },
  @{ src = "xebec.jpg"; dest = "." },
  @{ src = "xebec_chibi.png"; dest = "." },
  @{ src = "xebec_chibi_suzukisan.png"; dest = "." },
  @{ src = "Chibi_tamiru.png"; dest = "." },
  @{ src = "Chibi_suzuki_tamiru.png"; dest = "." },
  @{ src = "Chibi_raul.png"; dest = "." },
  @{ src = "Chibi_sukuzisan_raul.png"; dest = "." },
  @{ src = "Fonts"; dest = "Fonts" }
)

foreach ($item in $optionalData) {
  if (Test-Path $item.src) {
    $args += @("--add-data", "$($item.src);$($item.dest)")
  } else {
    Write-Host "警告: オプションアセットが存在しません: $($item.src)（スキップします）"
  }
}

# Forzar inclusión del binario compilado de kiwisolver (requerido por matplotlib)
$kiwiPyd = Join-Path $here "Lib\\site-packages\\kiwisolver\\_cext.cp313-win_amd64.pyd"
if (Test-Path $kiwiPyd) {
  $args += @("--add-binary", "$kiwiPyd;kiwisolver")
} else {
  Write-Host "警告: kiwisolver _cext .pyd が見つかりません: $kiwiPyd"
}

$args += @("bootstrap.py")

Write-Host "PyInstaller を実行中（onefile）..."
Write-Host "$python -m PyInstaller $($args -join ' ')"
& $python -m PyInstaller @args
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller が失敗しました（終了コード: $LASTEXITCODE）"
}

Write-Host ""
Write-Host "✅ ビルド完了。確認: dist\\$name.exe"


