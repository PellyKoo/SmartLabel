"""FastAPI 应用入口"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.web.routers import engine, tasks, preannotate, qualitycheck, video, image as image_router, filesystem
from src.web.task_manager import get_task_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化全局 TaskManager
    get_task_manager()
    logger.info("SmartLabel Web 服务已启动")
    yield
    # 关闭时释放所有引擎
    get_task_manager().engine_pool.release_all()
    logger.info("SmartLabel Web 服务已关闭")


app = FastAPI(
    title="SmartLabel API",
    description="AI 辅助标注与质检平台 Web 接口",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS（内网使用，允许同域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 静态资源 no-cache，避免浏览器缓存旧版 JS/CSS
@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path == "/" or path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# 注册 routers
app.include_router(engine.router)
app.include_router(tasks.router)
app.include_router(preannotate.router)
app.include_router(qualitycheck.router)
app.include_router(video.router)
app.include_router(image_router.router)
app.include_router(filesystem.router)

# 挂载静态文件
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    """前端单页应用入口"""
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return {"message": "SmartLabel API 已启动", "docs": "/docs"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    ico = STATIC_DIR / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico), media_type="image/x-icon")
    from fastapi.responses import Response
    return Response(status_code=204)   # No Content，不报 404


@app.get("/health")
def health():
    return {"status": "ok"}
