"""
Phase 4 CLI 验证脚本。

验证点：
1. 配置深合并（default.yaml + profile.yaml）
2. profile 覆盖 default 的字段
3. 缺失配置时的错误处理
4. CLI 命令注册齐全
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.commands import _deep_merge, load_config


def test_deep_merge():
    print("\n" + "=" * 60)
    print("测试 1: 深合并逻辑")
    print("=" * 60)

    base = {
        "engine": {
            "type": "vlm",
            "vlm": {"model_path": "default_model", "max_new_tokens": 256},
        },
        "runtime": {"num_io_workers": 4, "log_level": "INFO"},
        "task": {"classification": {"categories": []}},
    }
    override = {
        "engine": {
            "vlm": {"model_path": "custom_model"},  # 只覆盖 model_path
        },
        "runtime": {"log_level": "DEBUG"},  # 只覆盖 log_level
        "task": {"classification": {"categories": ["normal", "fatigue"]}},
    }

    merged = _deep_merge(base, override)

    # dict 深合并
    assert merged["engine"]["type"] == "vlm", "type 应保留"
    assert merged["engine"]["vlm"]["model_path"] == "custom_model", "model_path 应被覆盖"
    assert merged["engine"]["vlm"]["max_new_tokens"] == 256, "max_new_tokens 应保留"
    assert merged["runtime"]["num_io_workers"] == 4, "num_io_workers 应保留"
    assert merged["runtime"]["log_level"] == "DEBUG", "log_level 应被覆盖"

    # list 整体替换（不追加）
    assert merged["task"]["classification"]["categories"] == ["normal", "fatigue"], "list 应整体替换"

    print("✅ dict 递归合并正确")
    print("✅ list 整体替换（不追加）正确")
    print(f"  合并后 engine.vlm: {merged['engine']['vlm']}")
    print(f"  合并后 runtime: {merged['runtime']}")


def test_load_config_with_profile():
    print("\n" + "=" * 60)
    print("测试 2: load_config 合并 default + profile")
    print("=" * 60)

    # 写一个临时 profile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write("""
engine:
  type: vlm
  vlm:
    model_path: /tmp/fake_model
    max_new_tokens: 128

task:
  video:
    categories: [normal, fatigue]
    window_sec: 3.0
""")
        profile_path = f.name

    try:
        cfg = load_config(profile_path)

        # profile 覆盖
        assert cfg["engine"]["vlm"]["model_path"] == "/tmp/fake_model", "model_path 应被覆盖"
        assert cfg["engine"]["vlm"]["max_new_tokens"] == 128, "max_new_tokens 应被覆盖"
        assert cfg["task"]["video"]["window_sec"] == 3.0, "window_sec 应被覆盖"
        assert cfg["task"]["video"]["categories"] == ["normal", "fatigue"]

        # default 保留
        assert cfg["engine"]["vlm"]["quantization"] == "awq", "default 的 quantization 应保留"
        assert cfg["task"]["video"]["stride_sec"] == 2.5, "default 的 stride_sec 应保留"
        assert cfg["runtime"]["log_level"] == "INFO", "default 的 log_level 应保留"

        # prompts_dir 被注入
        assert "prompts_dir" in cfg["engine"], "engine.prompts_dir 应被注入"
        assert os.path.isabs(cfg["engine"]["prompts_dir"]), "prompts_dir 应为绝对路径"

        print("✅ profile 字段正确覆盖 default")
        print("✅ default 中未被 profile 覆盖的字段保留")
        print("✅ prompts_dir 自动注入为绝对路径")
        print(f"  engine.vlm.model_path = {cfg['engine']['vlm']['model_path']}")
        print(f"  engine.vlm.quantization = {cfg['engine']['vlm']['quantization']} (来自 default)")
        print(f"  engine.prompts_dir = {cfg['engine']['prompts_dir']}")
    finally:
        os.unlink(profile_path)


def test_cli_help():
    print("\n" + "=" * 60)
    print("测试 3: CLI --help 输出（子进程）")
    print("=" * 60)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    script = str(PROJECT_ROOT / "scripts" / "run_cli.py")
    result = subprocess.run(
        [sys.executable, script, "--help"],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert result.returncode == 0, f"--help 应返回 0，实际 {result.returncode}"

    expected_cmds = ["preannotate", "qualitycheck", "video-classify", "benchmark"]
    for cmd in expected_cmds:
        assert cmd in result.stdout, f"未在帮助中找到命令: {cmd}"
        print(f"✅ 命令已注册: {cmd}")


def test_missing_config():
    print("\n" + "=" * 60)
    print("测试 4: 配置文件不存在时正常报错")
    print("=" * 60)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    script = str(PROJECT_ROOT / "scripts" / "run_cli.py")

    result = subprocess.run(
        [sys.executable, script, "video-classify",
         "--config", "/nonexistent/path.yaml",
         "--video-dir", "/tmp/x",
         "--output-dir", "/tmp/y"],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert result.returncode != 0, "不存在的配置应返回非 0"
    print(f"✅ 返回码 {result.returncode}（非 0）")
    print(f"  stderr/stdout: {(result.stdout + result.stderr).strip()[:120]}")


def main():
    test_deep_merge()
    test_load_config_with_profile()
    test_cli_help()
    test_missing_config()

    print("\n" + "=" * 60)
    print("🎉 Phase 4 CLI 核心功能验证通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
