"""
验证 VLMEngine 的 bnb4/bnb8 量化分支代码路径正确。

不实际加载 7B 模型（太慢），仅检查：
1. config 接受 bnb4/bnb8 时，构造 BitsAndBytesConfig 参数正确
2. config 接受 awq/none 时，保持原行为（不引入 bnb）
3. bitsandbytes 能被 import（确认环境就绪）
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.vlm_engine import VLMEngine


def test_bnb_import():
    print("\n" + "=" * 60)
    print("测试 1: bitsandbytes 可用性")
    print("=" * 60)
    import bitsandbytes as bnb
    print(f"✅ bitsandbytes {bnb.__version__}")
    from transformers import BitsAndBytesConfig
    cfg = BitsAndBytesConfig(load_in_4bit=True)
    print(f"✅ BitsAndBytesConfig(load_in_4bit=True) 构造成功: {cfg}")


def test_bnb4_load_kwargs():
    """拦截 from_pretrained 检查传入 kwargs"""
    print("\n" + "=" * 60)
    print("测试 2: quantization=bnb4 时 load_kwargs 正确")
    print("=" * 60)

    engine = VLMEngine({
        "model_path": "/fake/path",
        "quantization": "bnb4",
        "device": "cuda:0",
        "torch_dtype": "float16",
    })

    captured_kwargs = {}

    fake_model = MagicMock()
    fake_processor = MagicMock()

    def _fake_from_pretrained(path, **kwargs):
        captured_kwargs.update(kwargs)
        return fake_model

    with patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained",
               side_effect=_fake_from_pretrained), \
         patch("transformers.AutoProcessor.from_pretrained",
               return_value=fake_processor):
        engine.load()

    assert "quantization_config" in captured_kwargs, "应注入 BitsAndBytesConfig"
    bnb_cfg = captured_kwargs["quantization_config"]
    assert bnb_cfg.load_in_4bit is True, "应为 4-bit"
    assert bnb_cfg.bnb_4bit_quant_type == "nf4", "应为 nf4"
    assert bnb_cfg.bnb_4bit_use_double_quant is True, "应启用 double quant"
    assert captured_kwargs["device_map"] == "auto", "bnb 应用 device_map=auto"
    print("✅ bnb4: 4-bit NF4 + double quant + device_map=auto")


def test_bnb8_load_kwargs():
    print("\n" + "=" * 60)
    print("测试 3: quantization=bnb8 时 load_kwargs 正确")
    print("=" * 60)

    engine = VLMEngine({
        "model_path": "/fake/path",
        "quantization": "bnb8",
    })
    captured = {}

    def _fake(path, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained",
               side_effect=_fake), \
         patch("transformers.AutoProcessor.from_pretrained",
               return_value=MagicMock()):
        engine.load()

    bnb_cfg = captured["quantization_config"]
    assert bnb_cfg.load_in_8bit is True, "应为 8-bit"
    print("✅ bnb8: 8-bit quantization 正确")


def test_awq_none_unchanged():
    """awq/none 路径不应引入 BitsAndBytesConfig"""
    print("\n" + "=" * 60)
    print("测试 4: quantization=none 时不注入 bnb 配置")
    print("=" * 60)

    engine = VLMEngine({
        "model_path": "/fake/path",
        "quantization": "none",
        "device": "cuda:0",
    })
    captured = {}

    def _fake(path, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained",
               side_effect=_fake), \
         patch("transformers.AutoProcessor.from_pretrained",
               return_value=MagicMock()):
        engine.load()

    assert "quantization_config" not in captured, "none 分支不该有 bnb 配置"
    assert captured["device_map"] == "cuda:0", "none 分支 device_map 应为具体设备"
    print("✅ none 分支保持原行为")


def main():
    test_bnb_import()
    test_bnb4_load_kwargs()
    test_bnb8_load_kwargs()
    test_awq_none_unchanged()
    print("\n" + "=" * 60)
    print("🎉 VLM 量化分支验证通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
