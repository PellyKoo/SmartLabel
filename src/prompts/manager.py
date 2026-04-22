import os
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptManager:
    """Prompt 模板管理器，从文件加载并渲染模板。"""

    def __init__(self, prompts_dir: str):
        """
        Args:
            prompts_dir: Prompt 模板文件目录路径
        """
        self._prompts_dir = prompts_dir
        self._cache: dict[str, str] = {}

    def load(self, name: str) -> str:
        """
        加载指定名称的 Prompt 模板。

        Args:
            name: 模板名称（不含扩展名），如 "classify_json"

        Returns:
            模板文本内容
        """
        if name in self._cache:
            return self._cache[name]

        file_path = os.path.join(self._prompts_dir, f"{name}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prompt 模板不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            template = f.read().strip()

        self._cache[name] = template
        logger.info(f"加载 Prompt 模板: {name} ({file_path})")
        return template

    def render(self, name: str, **kwargs) -> str:
        """
        加载并渲染 Prompt 模板。

        使用 Python str.format_map 进行变量替换。
        模板中用 {variable_name} 占位，双花括号 {{ }} 转义为字面花括号。

        Args:
            name: 模板名称
            **kwargs: 模板变量

        Returns:
            渲染后的 Prompt 文本
        """
        template = self.load(name)

        # 将 list 类型参数转为逗号分隔字符串
        rendered_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, (list, tuple)):
                rendered_kwargs[k] = "、".join(str(item) for item in v)
            else:
                rendered_kwargs[k] = v

        try:
            return template.format_map(rendered_kwargs)
        except KeyError as e:
            logger.warning(f"Prompt 模板 '{name}' 缺少变量: {e}，返回原始模板")
            return template

    def list_templates(self) -> list[str]:
        """列出所有可用的模板名称"""
        if not os.path.isdir(self._prompts_dir):
            return []
        return [
            os.path.splitext(f)[0]
            for f in os.listdir(self._prompts_dir)
            if f.endswith(".txt")
        ]
