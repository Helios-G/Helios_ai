import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from prepare_chexlocalize_masks import (
    decode_uncompressed_rle,
    prepare_dataset,
)


class PrepareChexlocalizeMasksTest(unittest.TestCase):
    def test_decode_uncompressed_rle_uses_coco_column_major_order(self):
        rle = {"size": [3, 3], "counts": [1, 2, 6]}

        mask = decode_uncompressed_rle(rle)

        self.assertEqual(mask.size, (3, 3))
        self.assertEqual(mask.getpixel((0, 0)), 0)
        self.assertEqual(mask.getpixel((0, 1)), 255)
        self.assertEqual(mask.getpixel((0, 2)), 255)
        self.assertEqual(mask.getpixel((1, 0)), 0)

    def test_prepare_dataset_unions_selected_pathologies(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "source_images"
            output_dir = root / "prepared"
            image_dir.mkdir()

            Image.new("RGB", (3, 3), color=(8, 8, 8)).save(image_dir / "case001.png")
            annotations = {
                "case001": {
                    "Lung Opacity": {"size": [3, 3], "counts": [1, 1, 7]},
                    "Pneumothorax": {"size": [3, 3], "counts": [4, 1, 4]},
                    "Cardiomegaly": {"size": [3, 3], "counts": [0, 0, 9]},
                }
            }
            annotation_path = root / "annotations.json"
            annotation_path.write_text(json.dumps(annotations), encoding="utf-8")

            stats = prepare_dataset(
                annotation_path=annotation_path,
                source_image_dir=image_dir,
                output_dir=output_dir,
                pathologies=["Lung Opacity", "Pneumothorax"],
                image_size=3,
            )

            self.assertEqual(stats["written"], 1)
            self.assertTrue((output_dir / "images" / "case001.png").exists())
            mask = Image.open(output_dir / "masks" / "case001.png").convert("L")
            self.assertEqual(mask.getpixel((0, 1)), 255)
            self.assertEqual(mask.getpixel((1, 1)), 255)
            self.assertEqual(mask.getpixel((2, 2)), 0)


if __name__ == "__main__":
    unittest.main()
