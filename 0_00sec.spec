# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['bootstrap.py'],
    pathex=[],
    binaries=[('C:\\Users\\xebec0176\\Desktop\\0.00sec\\.venv\\Lib\\\\site-packages\\\\kiwisolver\\\\_cext.cp313-win_amd64.pyd', 'kiwisolver')],
    datas=[('SlpashScreen_General.png', '.'), ('SplashScreen.scale-100.png', '.'), ('xebec_logo_88.png', '.'), ('loading.gif', '.'), ('xebec.ico', '.'), ('manifest.json', '.'), ('xebec.jpg', '.'), ('xebec_chibi.png', '.'), ('xebec_chibi_suzukisan.png', '.'), ('Chibi_tamiru.png', '.'), ('Chibi_suzuki_tamiru.png', '.'), ('Chibi_raul.png', '.'), ('Chibi_sukuzisan_raul.png', '.'), ('Fonts', 'Fonts')],
    hiddenimports=['0sec', 'kiwisolver._cext'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='0_00sec',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['xebec.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='0_00sec',
)
