"""Utilities for reading NEU-DET Pascal VOC XML annotations and drawing boxes.

This module visualizes dataset ground-truth annotations; it does not claim to run
an object detector. NEU-DET must be downloaded separately by the user.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

NEU_DET_CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
COLORS = [(0, 220, 120), (255, 190, 40), (70, 170, 255), (255, 90, 90), (190, 120, 255), (255, 100, 200)]


def parse_voc_annotation(xml_source):
    """Parse a Pascal VOC XML path or XML text into image metadata and boxes."""
    if isinstance(xml_source, Path):
        root = ET.parse(str(xml_source)).getroot()
    elif isinstance(xml_source, str) and not xml_source.lstrip().startswith("<") and Path(xml_source).is_file():
        root = ET.parse(xml_source).getroot()
    else:
        root = ET.fromstring(str(xml_source))
    size = root.find("size")
    width = int(size.findtext("width", "0")) if size is not None else 0
    height = int(size.findtext("height", "0")) if size is not None else 0
    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "unknown").strip().lower().replace(" ", "_")
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            xmin, ymin, xmax, ymax = [int(float(box.findtext(k, "0"))) for k in ("xmin", "ymin", "xmax", "ymax")]
        except ValueError:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append({"class": name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    return {"filename": root.findtext("filename", ""), "width": width, "height": height, "objects": objects}


def draw_ground_truth_boxes(image, annotation):
    """Return a copy of image with annotated ground-truth boxes and labels."""
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    class_to_color = {name: COLORS[i % len(COLORS)] for i, name in enumerate(NEU_DET_CLASSES)}
    for obj in annotation.get("objects", []):
        name = obj["class"]
        color = class_to_color.get(name, (255, 255, 0))
        box = (obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"])
        draw.rectangle(box, outline=color, width=max(2, round(min(result.size) / 150)))
        label = name.replace("_", " ")
        try:
            font = ImageFont.load_default()
            tb = draw.textbbox((box[0], box[1]), label, font=font)
            draw.rectangle((tb[0], tb[1], tb[2] + 4, tb[3] + 2), fill=color)
            draw.text((box[0] + 2, box[1]), label, fill=(0, 0, 0), font=font)
        except Exception:
            draw.text((box[0], box[1]), label, fill=color)
    return result


def find_annotation_pair(image_path, annotations_dir):
    """Find same-stem XML in a Pascal VOC annotations directory."""
    candidate = Path(annotations_dir) / (Path(image_path).stem + ".xml")
    return candidate if candidate.is_file() else None
