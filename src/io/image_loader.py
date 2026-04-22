import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 支持的图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def scan_images(image_dir: str, recursive: bool = False) -> list[str]:
    """
    扫描目录中的所有图片文件，按文件名排序。

    Args:
        image_dir: 图片目录路径
        recursive: 是否递归扫描子目录

    Returns:
        图片文件绝对路径列表
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    images = []
    if recursive:
        for root, _, files in os.walk(image_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    images.append(os.path.join(root, f))
    else:
        for f in os.listdir(image_dir):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(image_dir, f))

    images.sort()
    logger.info(f"扫描到 {len(images)} 张图片: {image_dir}")
    return images


def load_image(image_path: str, mode: str = "RGB") -> Optional[np.ndarray]:
    """
    加载单张图片。

    Args:
        image_path: 图片路径
        mode: "RGB" 或 "BGR"

    Returns:
        numpy 数组，加载失败返回 None
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"无法读取图片: {image_path}")
        return None
    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_images_batch(image_paths: list[str], mode: str = "RGB",
                       num_workers: int = 4) -> list[tuple[str, Optional[np.ndarray]]]:
    """
    多线程批量加载图片。

    Args:
        image_paths: 图片路径列表
        mode: "RGB" 或 "BGR"
        num_workers: IO 线程数

    Returns:
        [(path, image_array 或 None), ...]
    """
    def _load(path):
        return path, load_image(path, mode)

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(_load, image_paths):
            results.append(result)

    loaded_count = sum(1 for _, img in results if img is not None)
    logger.info(f"批量加载完成: {loaded_count}/{len(image_paths)} 张成功")
    return results
