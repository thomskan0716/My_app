$ErrorActionPreference = "SilentlyContinue"
$targets = @(
  "Lib\\site-packages\\PyQt5",
  "Lib\\site-packages\\PyQt6",
  "Lib\\site-packages\\PyQt5_Qt5-5.15.2.dist-info",
  "Lib\\site-packages\\PyQt5-5.15.11.dist-info",
  "Lib\\site-packages\\PyQt5_sip-12.17.0.dist-info",
  "Lib\\site-packages\\PyQt6_Qt6-6.9.0.dist-info",
  "Lib\\site-packages\\PyQt6-6.9.0.dist-info",
  "Lib\\site-packages\\PyQt6_sip-13.10.0.dist-info"
)
foreach($t in $targets){
  if(Test-Path $t){
    Remove-Item -Recurse -Force $t
    Write-Output "Deleted: $t"
  }
}
Write-Output "Remaining Qt bindings in site-packages:"
Get-ChildItem Lib\\site-packages | Where-Object { $_.Name -match '^(PyQt|PySide)' } | Select-Object Name,Mode | Format-Table -AutoSize | Out-String | Write-Output
