"""Train a small browser-friendly X-ray lesion segmentation model.

Expected input layout:

    dataset/
      images/
        study_a.png
        study_b.png
      masks/
        study_a.png
        study_b.png

Masks should be binary lesion masks. For CheXlocalize, prepare these masks by
unioning the pathology annotations you want to treat as "lesion".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import tensorflow as tf


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train X-ray lesion segmentation U-Net")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--validation-split", default=0.2, type=float)
    return parser.parse_args()


def list_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[str, str]]:
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    pairs: list[tuple[str, str]] = []
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if mask_path.exists():
            pairs.append((str(image_path), str(mask_path)))
    if not pairs:
        raise ValueError(f"No image/mask pairs found under {image_dir} and {mask_dir}")
    return pairs


def load_pair(image_path: tf.Tensor, mask_path: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0

    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.io.decode_image(mask_bytes, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, [image_size, image_size], method="nearest")
    mask = tf.cast(mask > 0, tf.float32)
    return image, mask


def make_dataset(
    pairs: Iterable[tuple[str, str]],
    image_size: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    image_paths, mask_paths = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), list(mask_paths)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda image_path, mask_path: load_pair(image_path, mask_path, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def conv_block(inputs: tf.Tensor, filters: int) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_unet(image_size: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    c1 = conv_block(inputs, 16)
    p1 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 32)
    p2 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 64)

    u2 = tf.keras.layers.UpSampling2D()(c3)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c4 = conv_block(u2, 32)
    u1 = tf.keras.layers.UpSampling2D()(c4)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c5 = conv_block(u1, 16)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c5)
    return tf.keras.Model(inputs, outputs, name="xray_lesion_unet")


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    smooth = tf.constant(1e-6, dtype=tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (denominator + smooth)


def iou_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    smooth = tf.constant(1e-6, dtype=tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred - (y_true * y_pred))
    return (intersection + smooth) / (union + smooth)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + (1.0 - dice_coefficient(y_true, y_pred))


def export_model(model: tf.keras.Model, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.export(output_dir)
    except AttributeError:
        model.save(output_dir, save_format="tf")


def main() -> None:
    args = parse_args()
    pairs = list_pairs(args.image_dir, args.mask_dir)
    split_index = max(1, int(len(pairs) * (1.0 - args.validation_split)))
    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:] or pairs[:1]

    train_ds = make_dataset(train_pairs, args.image_size, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_pairs, args.image_size, args.batch_size, shuffle=False)

    model = build_unet(args.image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=bce_dice_loss,
        metrics=[dice_coefficient, iou_coefficient],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    export_model(model, args.output_dir)
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
