# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — SmartLabel GUI 1.0.0

输出：build/dist/SmartLabel-1.0.0/  整个文件夹拷走即可运行
启动入口：SmartLabel.exe
"""
import os
import sys
from PyInstaller.utils.hooks import (
    collect_data_files, collect_submodules, collect_dynamic_libs,
)

# ---------- 路径 ----------
HERE = os.path.abspath(os.path.dirname(SPEC))   # build/
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))

# ---------- 数据文件（运行时需要的非 Python 文件）----------
datas = [
    (os.path.join(PROJECT_ROOT, "configs"),                           "configs"),
    (os.path.join(PROJECT_ROOT, "src", "gui", "styles", "theme.qss"),       "src/gui/styles"),
    (os.path.join(PROJECT_ROOT, "src", "gui", "styles", "theme_light.qss"), "src/gui/styles"),
    (os.path.join(PROJECT_ROOT, "src", "icon", "smartlabel_icon.ico"),      "src/icon"),
]
# transformers / tokenizers 会动态加载内置词表 / 配置
datas += collect_data_files("transformers", include_py_files=False)
datas += collect_data_files("torch",        include_py_files=False)
datas += collect_data_files("bitsandbytes", include_py_files=False)
datas += collect_data_files("qwen_vl_utils", include_py_files=False)

# ---------- 二进制（CUDA dll、bnb so/dll 等）----------
binaries  = collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("bitsandbytes")

# CUDA runtime DLL：PyInstaller 不会自动收集 conda 环境里的 CUDA 工具包
# 这些 DLL 是 c10_cuda.dll / torch_cuda.dll 的运行时依赖
# 放到 torch/lib/ 下，和 c10_cuda.dll 同级，确保 DLL 搜索路径能找到
import glob as _glob
_CONDA_PREFIX = sys.prefix  # conda env root（比 sys.executable 更可靠）
_cuda_patterns = [
    os.path.join(_CONDA_PREFIX, "bin",             "cublas*.dll"),
    os.path.join(_CONDA_PREFIX, "bin",             "cublasLt*.dll"),
    os.path.join(_CONDA_PREFIX, "bin",             "cudart*.dll"),
    os.path.join(_CONDA_PREFIX, "bin",             "cufft*.dll"),
    os.path.join(_CONDA_PREFIX, "bin",             "cusolver*.dll"),
    os.path.join(_CONDA_PREFIX, "bin",             "cusparse*.dll"),
    os.path.join(_CONDA_PREFIX, "bin",             "nvrtc*.dll"),
    os.path.join(_CONDA_PREFIX, "bin",             "nvJitLink*.dll"),
    os.path.join(_CONDA_PREFIX, "bin",             "nvblas*.dll"),
    os.path.join(_CONDA_PREFIX, "Library", "bin",  "curand*.dll"),
]
for pat in _cuda_patterns:
    for dll in _glob.glob(pat):
        binaries.append((dll, "torch/lib"))

# ---------- 隐藏导入（PyInstaller 静态分析抓不到的）----------
hiddenimports = [
    # 项目运行时按需 import（在 Tab 业务里 from src.pipeline.xxx import ...）
    "src.pipeline.preannotate",
    "src.pipeline.qualitycheck",
    "src.pipeline.video_classify",
    "src.report.generator",
    "src.utils.metrics",

    # transformers Qwen2-VL 系列（动态查找模型类）
    "transformers.models.qwen2_vl",
    "transformers.models.qwen2_vl.modeling_qwen2_vl",
    "transformers.models.qwen2_vl.configuration_qwen2_vl",
    "transformers.models.qwen2_vl.image_processing_qwen2_vl",
    "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast",
    "transformers.models.qwen2_vl.processing_qwen2_vl",
    "transformers.models.qwen2_vl.tokenization_qwen2",
    "transformers.models.qwen2_vl.tokenization_qwen2_fast",
    "transformers.models.qwen2",

    # 量化 / 加速
    "bitsandbytes",
    "bitsandbytes.nn",
    "bitsandbytes.optim",
    "accelerate",
    "qwen_vl_utils",

    # sklearn 内部 cython 模块（PyInstaller 偶尔漏）
    "sklearn.utils._cython_blas",
    "sklearn.tree._utils",
    "sklearn.neighbors._typedefs",
    "sklearn.neighbors._quad_tree",
]
# 把 transformers.models.qwen2_vl 的所有子模块都拉进来，避免运行时 ImportError
hiddenimports += collect_submodules("transformers.models.qwen2_vl")

# ---------- 减小体积 + 避免 GUI 模式炸 ----------
excludes = [
    # 项目其他前端
    "src.web", "src.cli",

    # 测试 / 文档
    "pytest", "_pytest",
    "IPython", "jupyter", "jupyter_core", "notebook",
    "tensorboard", "tensorboardX",
    "matplotlib.tests", "numpy.tests", "pandas.tests", "scipy.tests",

    # 不会用的可视化后端（PyQt5 已经够了）
    "tkinter", "tkinter.test",

    # 排除 nltk：被 scipy 间接拉进来，但项目用不到；
    # 它在 import 时会读 sys.stdout，windowed 模式下崩溃
    "nltk",
    # numpy.f2py：scipy 导入链需要，不能排除
    # scipy 子模块中 GUI 模式不会用到的
    "scipy.misc", "scipy.io.tests",
    # gensim/spacy/sklearn 测试：偶尔被拉进
    "gensim", "spacy",
]

# ---------- 分析 ----------
a = Analysis(
    [os.path.join(PROJECT_ROOT, "scripts", "run_gui.py")],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(HERE, "runtime_hook_streams.py")],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

# ---------- 主 EXE ----------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SmartLabel",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                  # UPX 压缩对 torch 这类大库收益小且容易触发 AV 误报
    console=False,              # GUI 应用不开控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(PROJECT_ROOT, "src", "icon", "smartlabel_icon.ico"),
    version=os.path.join(HERE, "version_info.txt"),
)

# ---------- COLLECT 输出整个 onedir ----------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SmartLabel-1.0.0",
)
