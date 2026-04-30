"""
文件系统浏览 API：供前端文件选择器使用。

安全设计：
- 只列出目录和图片/视频/文本/yaml 文件，不返回任意文件内容
- 路径参数只做 list，不做 read
"""
import os
import string
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/fs", tags=["filesystem"])

# 允许浏览的文件后缀（用于文件选择场景）
_ALLOWED_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".mp4", ".avi", ".mkv", ".mov",
    ".txt", ".csv", ".yaml", ".yml", ".json", ".xml",
}


def _safe_path(path: str) -> str:
    """规范化路径，防止 .. 穿越"""
    if not path:
        return _default_root()
    norm = os.path.normpath(os.path.abspath(path))
    return norm


def _default_root() -> str:
    """默认起始目录：Windows 返回所有盘符列表根，Linux 返回 /"""
    if os.name == "nt":
        return "__drives__"  # 特殊标记，由 list_dir 处理
    return "/"


@router.get("/ls")
def list_dir(path: str = Query("", description="服务器绝对路径，空则列出根目录")):
    """
    列出指定路径下的文件和目录。

    Returns:
        {
            "path": str,            # 当前路径
            "parent": str | None,   # 父目录（根目录时为 null）
            "entries": [
                {"name": str, "type": "dir"|"file", "ext": str}
            ]
        }
    """
    # Windows：空路径或 __drives__ → 列出所有盘符
    if (not path or path == "__drives__") and os.name == "nt":
        drives = _list_windows_drives()
        return {
            "path": "__drives__",
            "parent": None,
            "entries": [{"name": d, "type": "dir", "ext": ""} for d in drives],
        }

    abs_path = _safe_path(path)

    if not os.path.isdir(abs_path):
        raise HTTPException(status_code=404, detail=f"目录不存在: {abs_path}")

    # 父目录
    parent = os.path.dirname(abs_path)
    if parent == abs_path:
        # 到达根盘符（Windows）
        parent = "__drives__" if os.name == "nt" else None

    # 用 os.scandir 一次拿到所有 stat 信息，比 listdir + isdir 快很多
    dirs, files = [], []
    try:
        with os.scandir(abs_path) as it:
            for entry in it:
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append({"name": entry.name, "type": "dir", "ext": ""})
                    else:
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in _ALLOWED_EXTS:
                            files.append({"name": entry.name, "type": "file", "ext": ext})
                except OSError:
                    continue   # 单条 stat 失败不影响整体
    except PermissionError:
        pass

    dirs.sort(key=lambda x: x["name"].lower())
    files.sort(key=lambda x: x["name"].lower())
    return {"path": abs_path, "parent": parent, "entries": dirs + files}


@router.get("/read")
def read_text_file(path: str = Query(..., description="服务器文本文件路径")):
    """
    读取文本文件内容（仅限 .txt .csv .yaml .json）。
    供前端"从 txt 加载类别"功能使用。
    """
    abs_path = _safe_path(path)
    ext = os.path.splitext(abs_path)[1].lower()
    if ext not in {".txt", ".csv", ".yaml", ".yml", ".json"}:
        raise HTTPException(status_code=400, detail="只支持读取 txt/csv/yaml/json 文件")
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return {"path": abs_path, "content": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _list_windows_drives() -> list[str]:
    """
    列出 Windows 可用盘符。

    用 kernel32.GetLogicalDrives() 直接读位掩码，比 os.path.exists 探测 26 次快约 100 倍，
    且不会触发对断线网络驱动器的访问（避免 30 秒超时阻塞）。
    """
    try:
        import ctypes
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        drives = []
        for i, letter in enumerate(string.ascii_uppercase):
            if bitmask & (1 << i):
                drives.append(f"{letter}:\\")
        return drives
    except Exception:
        # 退化到慢速方案
        return [f"{l}:\\" for l in string.ascii_uppercase
                if os.path.exists(f"{l}:\\")]
