# SmartLabel GUI 使用说明（打包版）

## 快速开始

1. **解压**：把整个 `SmartLabel-1.0.0` 文件夹拷到任意位置（建议放在非中文路径）
2. **放模型**：把 Qwen2-VL 模型放到 `models/Qwen2-VL-2B-Instruct/` 子目录下
   （如果 models 目录不存在，自己新建一个）
3. **双击运行**：`SmartLabel.exe`

> 首次启动约 5-10 秒（解压 PyInstaller 引导器），之后启动 < 2 秒。

---

## GPU / CPU 自动适配

| 你的机器 | 自动选择 | 说明 |
|---|---|---|
| 有 NVIDIA GPU 且 CUDA 12.x 驱动 | **GPU 推理** | 7B 模型可用 bnb4 量化（~6.5GB 显存） |
| 无 GPU 或仅集显 | **CPU 推理** | **只能用 2B 模型**，7B 在 CPU 上会 OOM |

启动后在 GUI 左侧引擎面板加载模型，会在状态栏看到 `(device=cuda:0)` 或 `(device=cpu)`，确认是否走对了硬件路径。

---

## 模型获取

### 推荐：Qwen2-VL-2B-Instruct（约 4 GB）

```
modelscope download --model Qwen/Qwen2-VL-2B-Instruct \
    --local_dir ./models/Qwen2-VL-2B-Instruct
```

### GPU 用户可选：Qwen2-VL-7B-Instruct（约 16 GB）

```
modelscope download --model Qwen/Qwen2-VL-7B-Instruct \
    --local_dir ./models/Qwen2-VL-7B-Instruct
```

---

## 引擎面板配置示例

| 字段 | 推荐值 |
|---|---|
| 类型 | VLM (Qwen2-VL) |
| 模型路径 | `models\Qwen2-VL-2B-Instruct`（相对路径或绝对都行） |
| 量化 | **GPU 用户**：`bnb4`<br>**CPU 用户**：`none`（程序会自动降级） |

---

## 目录结构

```
SmartLabel-1.0.0/
├── SmartLabel.exe         ← 双击启动
├── _internal/             ← 所有依赖库（不要删！）
├── configs/               ← 配置文件
│   ├── default.yaml
│   ├── profiles/
│   └── prompts/
├── models/                ← 自己放模型到这里
│   └── Qwen2-VL-2B-Instruct/
└── 使用说明.md            ← 本文档
```

---

## 常见问题

### Q1：双击没反应？
A：在命令行用 `SmartLabel.exe` 启动，看具体报错。最常见是模型路径写错。

### Q2：报错 "CUDA out of memory"
A：显存不够，关掉其他占显存的程序；或者改用更小的模型（2B 而非 7B）；或者改用 `bnb4` 量化。

### Q3：CPU 模式特别慢？
A：正常。CPU 跑 2B 模型每张图约 5-10 秒，建议小批量数据用，大批量请上 GPU 机器。

### Q4：看到 "auto_engine_key 推荐用 bnb4 但 CPU 不支持，已切到 none"
A：正常。程序检测到 CPU 后自动调整，不会崩溃，正常使用即可。

### Q5：怎么改默认配置？
A：编辑 `configs/default.yaml`。改完重启 `SmartLabel.exe` 生效。

### Q6：杀毒软件报警 / Windows Defender 拦截？
A：PyInstaller 打包的 exe 偶尔被误报。把整个文件夹加白名单即可。本程序无任何网络上传行为，所有计算本地完成。

---

## 数据隐私

- 所有推理在**本地完成**，不向任何外部服务器发送数据
- 程序无网络请求（除非显式访问 ModelScope/HF 下载模型）
- 可在断网环境下运行

---

## 卸载

直接删除 `SmartLabel-1.0.0` 文件夹即可。注册表和系统目录均无残留。
