# PyInstaller spec file
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=collect_data_files('llama_cpp') + [
        ('Empty_Activity_android_studio_base_template', 'Empty_Activity_android_studio_base_template'),
    ],
    hiddenimports=['llama_cpp', 'huggingface_hub', 'requests', 'typer', 'ctransformers', 'gpt4all'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AndroidAgent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AndroidAgent'
)
