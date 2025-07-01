# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# Collect all data files and submodules for openhands
datas = []
binaries = []
hiddenimports = []

# Main packages to collect
packages = [
    'openhands',
    'aiohttp',
    'litellm',
    'docker',
    'pydantic',
    'toml',
    'rich',
    'prompt_toolkit',
    'websockets',
    'asyncio',
    'multidict',
    'yarl',
    'aiosignal',
    'frozenlist',
    'attrs',
    'charset_normalizer',
    'certifi',
    'idna',
    'urllib3',
    'requests',
    'pandas',
    'numpy',
    'typing_extensions',
    'jsonschema',
    'platformdirs',
    'packaging',
    'pyperclip',
    'wcwidth',
    'pygments',
    'markdown_it',
    'mdurl',
    'linkify_it_py',
    'mdit_py_plugins',
    'textual',
    'httpx',
    'httpcore',
    'h11',
    'anyio',
    'sniffio',
    'exceptiongroup',
    'sortedcontainers',
    'click',
    'python_dotenv',
    'aiohappyeyeballs',
]

for package in packages:
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
        datas.extend(package_datas)
        binaries.extend(package_binaries)
        hiddenimports.extend(package_hiddenimports)
    except Exception:
        # If collect_all fails, try to at least get submodules
        try:
            hiddenimports.extend(collect_submodules(package))
        except Exception:
            pass

# Additional hidden imports for openhands
hiddenimports.extend([
    'openhands.cli',
    'openhands.cli.main',
    'openhands.cli.commands',
    'openhands.cli.tui',
    'openhands.cli.settings',
    'openhands.cli.utils',
    'openhands.cli.suppress_warnings',
    'openhands.core',
    'openhands.core.config',
    'openhands.core.config.cli_config',
    'openhands.core.config.utils',
    'openhands.runtime',
    'openhands.controller',
    'openhands.server',
    'openhands.resolver',
    'openhands.llm',
    'openhands.memory',
    'openhands.storage',
    'openhands.agenthub',
    'openhands.utils',
    'openhands.events',
    'encodings.utf_8',
    'encodings.ascii',
    'encodings.latin_1',
    'asyncio.selector_events',
    'asyncio.base_events',
    'asyncio.protocols',
    'asyncio.transports',
    'asyncio.unix_events',
    'asyncio.windows_events' if sys.platform == 'win32' else '',
])

# Remove empty strings
hiddenimports = [h for h in hiddenimports if h]

# Entry point
a = Analysis(
    ['openhands/cli/main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'PIL',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'tkinter',
        'test',
        'tests',
        'pytest',
        'notebook',
        'jupyter',
        'IPython',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='openhands',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)