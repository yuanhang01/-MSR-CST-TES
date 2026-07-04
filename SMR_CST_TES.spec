# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置文件
用于将 SMR+CST+TES 仿真软件打包为单个 .exe 文件
"""
import os
import sys
sys.setrecursionlimit(5000)

# 收集 CoolProp 数据文件
import CoolProp
coolprop_path = os.path.dirname(CoolProp.__file__)

# 构建 datas 列表：收集 CoolProp 的物性数据文件
coolprop_datas = []
for root, dirs, files in os.walk(coolprop_path):
    for f in files:
        if f.endswith(('.json', '.h5', '.dat', '.csv', '.hdf5')):
            src = os.path.join(root, f)
            dst = os.path.join('CoolProp', os.path.relpath(root, coolprop_path))
            coolprop_datas.append((src, dst))

a = Analysis(
    ['gui_app.py'],
    pathex=[os.path.dirname(os.path.abspath(SPECPATH))],
    binaries=[],
    datas=[
        ('config.yaml', '.'),
    ] + coolprop_datas,
    hiddenimports=[
        'CoolProp',
        'CoolProp.CoolProp',
        'CoolProp.Plots',
        'CoolProp.State',
        'numpy',
        'numpy._core',
        'numpy.linalg',
        'numpy.fft',
        'scipy',
        'scipy.special',
        'scipy.optimize',
        'matplotlib',
        'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_tkagg',
        'pandas',
        'yaml',
        'tkinter',
        'tkinter.ttk',
        'tkinter.scrolledtext',
        'tkinter.messagebox',
        'PIL',
        'PIL.Image',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'jupyter',
        'notebook',
        'sphinx',
        'pytest',
        'setuptools',
        'pip',
        'wheel',
        'tkinter.test',
        'test',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SMR_CST_TES_Simulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,       # 不显示控制台窗口 (GUI 应用)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)