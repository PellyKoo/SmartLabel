# SmartLabel — AI 辅助标注与质检平台

> 通用图像 / 视频标注辅助工具，支持 **VLM + 自有模型双引擎**，覆盖预标注、质检、视频片段分类全流程，提供 PyQt5 桌面端、CLI 命令行、FastAPI Web 三种前端。

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [环境依赖](#环境依赖)
- [快速上手](#快速上手)
  - [安装](#安装)
  - [下载模型](#下载模型)
  - [运行 Demo](#运行-demo)
- [三种前端使用方式](#三种前端使用方式)
  - [PyQt5 桌面 GUI](#pyqt5-桌面-gui)
  - [CLI 命令行](#cli-命令行)
  - [FastAPI Web 服务](#fastapi-web-服务)
- [典型使用场景](#典型使用场景)
  - [场景一：DMS 大批量图片预标注](#场景一dms-大批量图片预标注)
  - [场景二：外包标注质检](#场景二外包标注质检)
  - [场景三：视频片段自动分类](#场景三视频片段自动分类)
  - [场景四：基准测试评估](#场景四基准测试评估)
- [配置说明](#配置说明)
- [目录结构](#目录结构)

---

## 项目概述

### 解决的痛点

| 阶段 | 痛点 | SmartLabel 方案 |
|------|------|-----------------|
| 成熟项目（有训练模型） | 外包标注质量不稳定 | 自有模型初筛 + VLM 低置信度升级复核 |
| 新项目（无模型） | 标注从零开始，效率低 | VLM 通用预标注 |
| 视频数据 | 人工逐帧标注，极低效 | 滑动窗口 + 时序平滑 + 片段检测 |
| 跨平台部署 | Windows 本地 + Linux 服务器 | 桌面 GUI / CLI / Web 三端统一后端 |

### 支持的任务

- **图片分类预标注**：将大量无标注图片按类别自动分类
- **目标检测预标注**：自动框出目标，输出 Pascal VOC XML
- **分类标注质检**：对比人工标注与引擎判断，生成异议率报告
- **检测标注质检**：IoU 匹配，自动识别漏标 / 多标 / 类别错误
- **视频片段分类**：滑动窗口抽帧 → 逐窗口推理 → 时序平滑 → 输出时间段标签

---

## 系统架构

```
推理引擎（可插拔）：
  ├── VLM 引擎（Qwen2-VL）        → 通用：分类、质检推理、视频多帧理解
  └── 自有模型引擎                  → 专用：高精度分类、检测
       ├── PyTorch (.pt/.pth)
       ├── ONNX (.onnx)
       └── TensorRT (.engine)

前端（共享后端核心）：
  ├── PyQt5 桌面端     → Windows/Linux，全功能 GUI
  ├── CLI 终端         → Linux 服务器，批量无人值守
  └── Web（FastAPI）   → 服务器远程，浏览器操作
```

---

## 环境依赖

### 基础要求

| 项目 | 要求 |
|------|------|
| Python | 3.9+ |
| CUDA | 11.8 / 12.x |
| GPU 显存 | 最低 8GB（Qwen2-VL-7B bnb4 约 6.5GB） |
| OS | Windows 10/11 / Linux |

### 安装依赖

```bash
# 核心依赖
pip install torch>=2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.45.0 accelerate qwen-vl-utils
pip install bitsandbytes>=0.43.0      # Windows 4-bit 量化（替代 autoawq）
pip install fastapi uvicorn PyQt5 typer rich
pip install opencv-python lxml pyyaml scikit-learn matplotlib jinja2
```

或一键安装：

```bash
pip install -r requirements.txt
```

> **Windows 注意**：`autoawq` 依赖 `triton`，Windows 无官方 wheel。  
> 推荐使用 `quantization: bnb4`（bitsandbytes，已内置支持）代替 AWQ。

---

## 快速上手

### 安装

```bash
git clone <repo>
cd SmartLabel
pip install -r requirements.txt
```

### 下载模型

```bash
# 推荐：Qwen2-VL-7B（bnb4 量化后约 6.5GB 显存）
modelscope download --model Qwen/Qwen2-VL-7B-Instruct \
    --local_dir ./models/Qwen2-VL-7B-Instruct

# 轻量版（约 5GB 显存，精度较低）
modelscope download --model Qwen/Qwen2-VL-2B-Instruct \
    --local_dir ./models/Qwen2-VL-2B-Instruct
```

### 运行 Demo

无需 GPU，用 Mock 引擎验证完整流程（约 3 秒跑完）：

```bash
python scripts/demo_dms.py --dry-run
```

真实 VLM 运行：

```bash
python scripts/demo_dms.py \
    --model-path models/Qwen2-VL-7B-Instruct \
    --quantization bnb4 \
    --output-dir demo_output
```

输出物：
- `demo_output/preannotate/preannotate_report.html`  — 预标注汇总报告
- `demo_output/qualitycheck/qc_report.html`           — 质检 HTML 报告
- `demo_output/video/video_report.html`               — 视频时间轴报告

---

## 三种前端使用方式

### PyQt5 桌面 GUI

```bash
python scripts/run_gui.py
```

**界面布局**：左侧引擎面板 → 右侧 5 个 Tab（预标注 / 质检 / 视频分类 / 评估 / 设置）→ 底部日志

**使用步骤**：

1. **加载引擎**（左侧面板）  
   - 填写引擎 Key（任意名称，如 `vlm_7b`）  
   - 选择类型（VLM / ONNX / PyTorch）  
   - 填写模型路径  
   - 点击「加载」  

2. **预标注 Tab**  
   - 选择图片目录（选完自动预览待标注图片）  
   - 点击 📁 按钮从 txt 文件加载类别（每行一个）  
   - 选择输出目录 → 点击「开始预标注」  
   - 完成后可在右侧浏览器直接修改标签（自动移动文件）  

3. **质检 Tab**  
   - 填写图片目录、标注目录、输出目录、类别  
   - 可选：加载 VLM 辅助引擎开启低置信度升级策略  
   - 完成后导出 HTML / CSV 报告、错误 case  

4. **视频分类 Tab**  
   - 选择视频文件或目录  
   - 配置窗口时长、步长、平滑参数  
   - 运行后在右侧时间轴点击跳转视频播放位置  

5. **评估 Tab**  
   - 需要真值标注目录  
   - 运行后显示 Accuracy / F1 / 混淆矩阵图  

6. **设置 Tab**  
   - 调整运行参数（IO 线程数、日志级别）  
   - 直接编辑 Prompt 模板并保存  

---

### CLI 命令行

```bash
# 查看所有命令
python scripts/run_cli.py --help

# ---- 预标注 ----
# 分类预标注（使用 DMS 预设配置）
python scripts/run_cli.py preannotate \
    --config configs/profiles/dms_preannotate.yaml \
    --image-dir /data/dms/images \
    --output-dir /data/dms/pre_annotations \
    --task classification

# 从 txt 文件传入类别（每行一个类别）
python scripts/run_cli.py preannotate \
    --config configs/profiles/dms_preannotate.yaml \
    --image-dir /data/images \
    --output-dir /data/output \
    --categories normal,fatigue,distracted,phone,smoke

# 检测预标注（需要支持检测的自有模型）
python scripts/run_cli.py preannotate \
    --config configs/profiles/dms_preannotate.yaml \
    --image-dir /data/images \
    --output-dir /data/det_output \
    --task detection \
    --categories person,car

# ---- 质检 ----
# 分类质检（单引擎）
python scripts/run_cli.py qualitycheck \
    --config configs/profiles/dms_qualitycheck.yaml \
    --image-dir /data/dms/images \
    --annotation-dir /data/dms/annotations \
    --output-dir /data/dms/qc_report \
    --task classification

# 分类质检（双引擎：主引擎初筛 + VLM 升级复核）
python scripts/run_cli.py qualitycheck \
    --config configs/profiles/dms_qualitycheck.yaml \
    --image-dir /data/dms/images \
    --annotation-dir /data/dms/annotations \
    --output-dir /data/dms/qc_report \
    --vlm-config configs/profiles/dms_preannotate.yaml

# ---- 视频分类 ----
python scripts/run_cli.py video-classify \
    --config configs/profiles/dms_preannotate.yaml \
    --video-dir /data/dms/videos \
    --output-dir /data/dms/video_labels \
    --strategy temporal_smooth \
    --categories normal,fatigue,distracted,phone,smoke

# ---- 基准评估 ----
python scripts/run_cli.py benchmark \
    --config configs/profiles/dms_preannotate.yaml \
    --image-dir /data/benchmark/images \
    --gt-dir /data/benchmark/annotations \
    --output-dir /data/benchmark/results \
    --task classification
```

---

### FastAPI Web 服务

```bash
# 本机访问（默认 127.0.0.1:8000）
python scripts/run_web.py

# 内网访问
python scripts/run_web.py --host 0.0.0.0 --port 8000

# 开发模式（代码变更自动重载）
python scripts/run_web.py --reload
```

**访问地址**：
- 前端界面：`http://localhost:8000`
- API 文档（Swagger）：`http://localhost:8000/docs`

**API 快速参考**：

```bash
# 加载引擎
curl -X POST http://localhost:8000/api/engine/load \
  -H "Content-Type: application/json" \
  -d '{"engine_key":"vlm_7b","engine_config":{"type":"vlm","vlm":{"model_path":"models/Qwen2-VL-7B-Instruct","quantization":"bnb4"}}}'

# 提交预标注任务
curl -X POST http://localhost:8000/api/preannotate/classification/start \
  -H "Content-Type: application/json" \
  -d '{"engine_key":"vlm_7b","engine_config":{...},"image_dir":"/data/images","output_dir":"/data/out","categories":["normal","fatigue"]}'

# 查询任务状态
curl http://localhost:8000/api/tasks/<task_id>/status

# 获取任务结果
curl http://localhost:8000/api/tasks/<task_id>/result
```

---

## 典型使用场景

### 场景一：DMS 大批量图片预标注

**适用**：有大量无标注图片，用 VLM 做初步分类

```bash
python scripts/run_cli.py preannotate \
    --config configs/profiles/dms_preannotate.yaml \
    --image-dir /data/dms/raw_images \
    --output-dir /data/dms/pre_annotations
```

**输出**：
```
/data/dms/pre_annotations/
├── normal/          ← 按类别分文件夹
├── fatigue/
├── distracted/
├── phone/
├── smoke/
└── classification_results.csv
```

**类别 txt 文件格式**（`configs/dms_categories.txt`）：

```
# DMS 五分类
normal
fatigue
distracted
phone
smoke
```

---

### 场景二：外包标注质检

**适用**：外包标注完成，用引擎自动检查标注质量

```bash
python scripts/run_cli.py qualitycheck \
    --config configs/profiles/dms_qualitycheck.yaml \
    --image-dir /data/dms/images \
    --annotation-dir /data/dms/annotations \
    --output-dir /data/dms/qc_report \
    --vlm-config configs/profiles/dms_preannotate.yaml
```

**输出报告内容**：
- 总体通过率 / 异议率 / VLM 升级统计
- 各类别异议率柱状图
- 待复核样本图库（含 VLM 给出的理由）
- 高频错误模式（如"fatigue 被标为 normal"出现 N 次）
- 错误 case 分文件夹导出（供标注员复训）

**低置信度升级策略**：

```
自有模型分类 → confidence ≥ 0.8 → 直接判定
                confidence < 0.8 → VLM 深度复核（给出文字理由）
```

---

### 场景三：视频片段自动分类

**适用**：驾驶视频数据，自动输出每个时间段的行为标签

```bash
python scripts/run_cli.py video-classify \
    --config configs/profiles/dms_preannotate.yaml \
    --video-dir /data/dms/videos \
    --output-dir /data/dms/video_labels \
    --strategy temporal_smooth
```

**流程**：
```
视频 → 滑动窗口抽帧（5s窗口/2.5s步长）
     → 逐窗口 VLM 推理
     → 时序中值滤波（消除孤立噪声）
     → 片段合并（连续同类 ≥ 2s 合并）
     → 输出 CSV + JSON + HTML 时间轴报告
```

**输出**（`video_result.json`）：
```json
{
  "video_path": "/data/dms/videos/clip_001.mp4",
  "clips": [
    {"start_sec": 0.0, "end_sec": 12.5, "label": "normal", "duration_sec": 12.5},
    {"start_sec": 12.5, "end_sec": 18.0, "label": "phone", "duration_sec": 5.5},
    {"start_sec": 18.0, "end_sec": 35.0, "label": "normal", "duration_sec": 17.0}
  ],
  "statistics": {"normal": 29.5, "phone": 5.5}
}
```

---

### 场景四：基准测试评估

**适用**：有已标注真值数据，评估引擎在当前任务上的精度

```bash
# 快速验证（dry-run，Mock 引擎）
python scripts/run_benchmark.py --dry-run \
    --image-dir /data/benchmark/images \
    --gt-dir /data/benchmark/annotations \
    --categories normal,fatigue,distracted,phone,smoke

# 真实评估
python scripts/run_benchmark.py \
    --config configs/profiles/dms_preannotate.yaml \
    --image-dir /data/benchmark/images \
    --gt-dir /data/benchmark/annotations \
    --output-dir /data/benchmark/results
```

**输出**：
- 控制台打印 Accuracy / Macro-F1 / per-class 指标
- `benchmark_report.json` — 完整指标
- `confusion_matrix.png` — 混淆矩阵热力图

---

## 配置说明

### 引擎配置（`configs/default.yaml`）

```yaml
engine:
  type: vlm          # vlm | pytorch | onnx | tensorrt
  vlm:
    model_path: models/Qwen2-VL-7B-Instruct
    quantization: bnb4   # bnb4（Windows）| awq（Linux）| none（fp16）
    device: cuda:0
    torch_dtype: float16
    max_new_tokens: 128
    multi_sample: true   # 开启多次采样（精度↑ 速度↓）
    sample_count: 3
    temperature: 0.6
```

### 量化方式对比

| 量化 | 显存（7B）| 推理速度 | 平台 |
|------|----------|----------|------|
| `bnb4` | ~6.5 GB | ★★★★ | Windows / Linux |
| `bnb8` | ~9 GB | ★★★☆ | Windows / Linux |
| `awq` | ~5.5 GB | ★★★★★ | Linux（需 autoawq ≥ 0.2.5）|
| `none` | ~16 GB | ★★★★★ | 需 16GB 以上显存 |

### 预设 Profile

| Profile | 适用场景 |
|---------|---------|
| `configs/profiles/dms_preannotate.yaml` | DMS 驾驶员行为分类预标注 |
| `configs/profiles/dms_qualitycheck.yaml` | DMS 外包标注质检 |
| `configs/profiles/new_project_vlm.yaml` | 新项目从零开始 VLM 预标注 |

### 类别 txt 文件

各业务 Tab 和 CLI 均支持从 txt 文件加载类别，格式：

```
# 注释行（以 # 开头，自动忽略）
# 空行也会被忽略

normal
fatigue
distracted
phone
smoke
```

---

## 目录结构

```
SmartLabel/
├── configs/
│   ├── default.yaml              默认配置
│   ├── profiles/                 场景预设配置
│   │   ├── dms_preannotate.yaml
│   │   ├── dms_qualitycheck.yaml
│   │   └── new_project_vlm.yaml
│   └── prompts/                  Prompt 模板
│       ├── classify_json.txt
│       ├── qc_json.txt
│       └── video_clip.txt
├── models/                       模型文件（需自行下载）
├── scripts/
│   ├── run_gui.py               PyQt5 GUI 入口
│   ├── run_cli.py               CLI 入口
│   ├── run_web.py               FastAPI Web 入口
│   ├── demo_dms.py              DMS 演示脚本
│   └── run_benchmark.py         基准测试脚本
├── src/
│   ├── engine/                  推理引擎（VLM / 自有模型）
│   ├── pipeline/                业务流水线（预标注 / 质检 / 视频）
│   ├── io/                      数据 I/O（VOC XML / 分类 CSV / 视频）
│   ├── postprocess/             后处理（VLM 解析 / 时序平滑）
│   ├── prompts/                 Prompt 管理
│   ├── report/                  报告生成（HTML / CSV）
│   ├── utils/                   日志 / 评估指标
│   ├── gui/                     PyQt5 桌面端
│   ├── cli/                     CLI 命令（Typer）
│   └── web/                     FastAPI Web 服务
├── tests/                       单元测试 / 冒烟测试
├── requirements.txt
└── README.md
```

---

## 数据安全

| 措施 | 说明 |
|------|------|
| 全程内网 | 不依赖外部 API，可断网运行 |
| 零数据外传 | 所有推理在本地 GPU 完成 |
| Web 内网绑定 | 默认绑定 `127.0.0.1`，内网访问需显式 `--host 0.0.0.0` |
| 图片代理白名单 | `/api/image` 端点有路径越权检查 |
| 临时文件清理 | 视频抽帧临时文件处理完自动删除 |
