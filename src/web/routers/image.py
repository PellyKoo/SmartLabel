"""图片代理 API：安全地将服务器本地图片暴露给前端。"""
import os
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api", tags=["image"])

# 允许代理的根目录白名单（在 app.py 启动时写入）
_allowed_roots: list[str] = []


def set_allowed_roots(roots: list[str]):
    global _allowed_roots
    _allowed_roots = [os.path.abspath(r) for r in roots if os.path.isdir(r)]


def _is_allowed(path: str) -> bool:
    """检查路径是否在白名单根目录内，防止路径穿越"""
    abs_path = os.path.abspath(path)
    if not _allowed_roots:
        # 未配置白名单时，允许所有（开发模式）
        return True
    return any(abs_path.startswith(root) for root in _allowed_roots)


@router.get("/image")
def serve_image(path: str = Query(..., description="服务器本地图片绝对路径")):
    """
    图片代理端点：前端用 /api/image?path=xxx 展示服务器图片。

    安全：路径必须在 allowed_roots 白名单内。
    """
    if not _is_allowed(path):
        raise HTTPException(status_code=403, detail="路径不在允许范围内")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="图片不存在")
    suffix = os.path.splitext(path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp",
        ".webp": "image/webp", ".gif": "image/gif",
    }
    media_type = media_type_map.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media_type)
