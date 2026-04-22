import os
from typing import Optional
from lxml import etree

from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_voc_xml(xml_path: str) -> dict:
    """
    解析 Pascal VOC XML 标注文件。

    Returns:
        {
            "filename": str,
            "size": {"width": int, "height": int, "depth": int},
            "objects": [
                {"label": str, "bbox": [x1, y1, x2, y2], "difficult": bool},
                ...
            ]
        }
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename", "")
    size_elem = root.find("size")
    size = {
        "width": int(size_elem.findtext("width", "0")),
        "height": int(size_elem.findtext("height", "0")),
        "depth": int(size_elem.findtext("depth", "3")),
    } if size_elem is not None else {"width": 0, "height": 0, "depth": 3}

    objects = []
    for obj in root.findall("object"):
        label = obj.findtext("name", "")
        difficult = obj.findtext("difficult", "0") == "1"
        bbox_elem = obj.find("bndbox")
        if bbox_elem is None:
            continue
        bbox = [
            int(float(bbox_elem.findtext("xmin", "0"))),
            int(float(bbox_elem.findtext("ymin", "0"))),
            int(float(bbox_elem.findtext("xmax", "0"))),
            int(float(bbox_elem.findtext("ymax", "0"))),
        ]
        objects.append({"label": label, "bbox": bbox, "difficult": difficult})

    return {"filename": filename, "size": size, "objects": objects}


def write_voc_xml(output_path: str, filename: str,
                   img_width: int, img_height: int,
                   detections: list[dict],
                   img_depth: int = 3,
                   folder: str = ""):
    """
    写入 Pascal VOC XML 标注文件。

    Args:
        output_path: 输出 XML 路径
        filename: 图片文件名
        img_width: 图片宽度
        img_height: 图片高度
        detections: [{"label": str, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
        img_depth: 图片通道数
        folder: 文件夹名（可选）
    """
    root = etree.Element("annotation")

    etree.SubElement(root, "folder").text = folder
    etree.SubElement(root, "filename").text = filename

    size = etree.SubElement(root, "size")
    etree.SubElement(size, "width").text = str(img_width)
    etree.SubElement(size, "height").text = str(img_height)
    etree.SubElement(size, "depth").text = str(img_depth)

    for det in detections:
        obj = etree.SubElement(root, "object")
        etree.SubElement(obj, "name").text = det["label"]
        etree.SubElement(obj, "pose").text = "Unspecified"
        etree.SubElement(obj, "truncated").text = "0"
        etree.SubElement(obj, "difficult").text = "0"

        if det.get("confidence") is not None:
            etree.SubElement(obj, "confidence").text = str(round(det["confidence"], 4))

        bbox = det["bbox"]
        bndbox = etree.SubElement(obj, "bndbox")
        etree.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
        etree.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
        etree.SubElement(bndbox, "xmax").text = str(int(bbox[2]))
        etree.SubElement(bndbox, "ymax").text = str(int(bbox[3]))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree = etree.ElementTree(root)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="utf-8")


def read_voc_annotations(xml_dir: str) -> dict[str, dict]:
    """
    批量读取目录下所有 VOC XML 标注。

    Returns:
        {filename: parsed_annotation_dict, ...}
    """
    if not os.path.isdir(xml_dir):
        raise FileNotFoundError(f"XML 标注目录不存在: {xml_dir}")

    annotations = {}
    for f in os.listdir(xml_dir):
        if not f.endswith(".xml"):
            continue
        xml_path = os.path.join(xml_dir, f)
        try:
            parsed = parse_voc_xml(xml_path)
            key = parsed["filename"] or os.path.splitext(f)[0]
            annotations[key] = parsed
        except Exception as e:
            logger.warning(f"解析 XML 失败: {xml_path}, 错误: {e}")

    logger.info(f"读取 VOC 标注: {len(annotations)} 条，来自 {xml_dir}")
    return annotations
