"""Prepare CheXlocalize-style annotations for HELIOS lesion segmentation.

The training script expects paired image and binary mask PNG files:

    output/
      images/<case_id>.png
      masks/<case_id>.png

This script reads a CheXlocalize-style JSON file, unions selected pathology
segmentations into a single binary lesion mask, and copies/resizes matching
source images into the same naming scheme.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops


DEFAULT_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumothorax",
    "Support Devices",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CheXlocalize binary lesion masks")
    parser.add_argument("--annotation-json", required=True, type=Path)
    parser.add_argument("--source-image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument(
        "--pathology",
        action="append",
        dest="pathologies",
        help="Pathology to include. Repeat to include multiple. Defaults to CheXlocalize lesion pathologies.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write all-zero masks when a case has no selected pathology annotation.",
    )
    return parser.parse_args()


def sanitize_case_id(case_id: str) -> str:
    value = Path(str(case_id)).stem
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "case"


def decode_uncompressed_rle(rle: dict[str, Any]) -> Image.Image:
    """Decode COCO uncompressed RLE into a binary PIL mask.

    COCO RLE is stored in column-major order. Counts alternate background and
    foreground runs, starting with background.
    """

    height, width = [int(value) for value in rle["size"]]
    counts = [int(value) for value in rle["counts"]]
    pixels = [0] * (height * width)
    cursor = 0
    value = 0
    for run_length in counts:
        for offset in range(cursor, min(cursor + run_length, len(pixels))):
            pixels[offset] = value
        cursor += run_length
        value = 255 if value == 0 else 0

    mask = Image.new("L", (width, height), 0)
    for y in range(height):
        for x in range(width):
            mask.putpixel((x, y), pixels[x * height + y])
    return mask


def decode_compressed_rle(rle: dict[str, Any]) -> Image.Image:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError(
            "Compressed CheXlocalize RLE requires pycocotools. "
            "Install it with `pip install pycocotools` in the training environment."
        ) from exc

    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.max(axis=2)
    height, width = decoded.shape[:2]
    return Image.fromarray((decoded > 0).astype("uint8") * 255, mode="L").resize(
        (width, height),
        Image.Resampling.NEAREST,
    )


def decode_rle(rle: Any) -> Image.Image | None:
    if not rle:
        return None
    if isinstance(rle, list):
        masks = [decode_rle(item) for item in rle]
        masks = [mask for mask in masks if mask is not None]
        return union_masks(masks)
    if not isinstance(rle, dict):
        return None
    counts = rle.get("counts")
    if counts is None or "size" not in rle:
        return None
    if isinstance(counts, list):
        return decode_uncompressed_rle(rle)
    return decode_compressed_rle(rle)


def union_masks(masks: Iterable[Image.Image]) -> Image.Image | None:
    masks = list(masks)
    if not masks:
        return None
    base = masks[0].convert("L")
    for mask in masks[1:]:
        if mask.size != base.size:
            mask = mask.resize(base.size, Image.Resampling.NEAREST)
        base = ImageChops.lighter(base, mask.convert("L"))
    return base.point(lambda value: 255 if value > 0 else 0)


def read_annotation_records(annotation_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and all(isinstance(value, dict) for value in payload.values()):
        return payload
    if isinstance(payload, list):
        records: dict[str, dict[str, Any]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            case_id = str(item.get("id") or item.get("image_id") or item.get("study_id") or "")
            segmentations = item.get("segmentations") or item.get("annotations") or {}
            if case_id and isinstance(segmentations, dict):
                records[case_id] = segmentations
        if records:
            return records
    raise ValueError("Unsupported annotation JSON shape. Expected {case_id: {pathology: rle}}.")


def build_image_index(source_image_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in source_image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.name] = path
            index[path.stem] = path
            index[sanitize_case_id(path.stem)] = path
    return index


def find_source_image(case_id: str, image_index: dict[str, Path]) -> Path | None:
    candidates = [
        str(case_id),
        Path(str(case_id)).name,
        Path(str(case_id)).stem,
        sanitize_case_id(str(case_id)),
    ]
    for candidate in candidates:
        if candidate in image_index:
            return image_index[candidate]
    return None


def write_image(source_path: Path, output_path: Path, image_size: int) -> None:
    image = Image.open(source_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_mask(mask: Image.Image, output_path: Path, image_size: int) -> None:
    mask = mask.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    mask = mask.point(lambda value: 255 if value > 0 else 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)


def prepare_dataset(
    annotation_path: Path,
    source_image_dir: Path,
    output_dir: Path,
    pathologies: list[str] | None = None,
    image_size: int = 256,
    allow_empty: bool = False,
) -> dict[str, int]:
    records = read_annotation_records(annotation_path)
    selected_pathologies = pathologies or DEFAULT_PATHOLOGIES
    image_index = build_image_index(source_image_dir)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    written = 0
    skipped_missing_image = 0
    skipped_empty_mask = 0

    for case_id, pathology_map in records.items():
        source_image = find_source_image(case_id, image_index)
        if source_image is None:
            skipped_missing_image += 1
            continue

        masks = []
        for pathology in selected_pathologies:
            mask = decode_rle(pathology_map.get(pathology))
            if mask is not None:
                masks.append(mask)
        union_mask = union_masks(masks)
        if union_mask is None:
            if not allow_empty:
                skipped_empty_mask += 1
                continue
            with Image.open(source_image) as image:
                union_mask = Image.new("L", image.size, 0)

        output_name = f"{sanitize_case_id(case_id)}.png"
        write_image(source_image, images_dir / output_name, image_size)
        write_mask(union_mask, masks_dir / output_name, image_size)
        written += 1

    stats = {
        "records": len(records),
        "written": written,
        "skipped_missing_image": skipped_missing_image,
        "skipped_empty_mask": skipped_empty_mask,
    }
    return stats


def main() -> None:
    args = parse_args()
    stats = prepare_dataset(
        annotation_path=args.annotation_json,
        source_image_dir=args.source_image_dir,
        output_dir=args.output_dir,
        pathologies=args.pathologies,
        image_size=args.image_size,
        allow_empty=args.allow_empty,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
