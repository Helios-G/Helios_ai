# X-ray Lesion Segmentation Model Pipeline

This folder contains the local-only model preparation path for HELIOS X-ray
lesion segmentation.

## 1. Prepare Image/Mask Pairs

Download CheXlocalize from Stanford AIMI and keep the data outside git. The
annotation JSON should map case ids to pathology RLE masks.

```bash
cd /Users/kimseonmin/HELIOS/new/HELIOS-Main/helios_ai

python3 preprocessing/segmentation/prepare_chexlocalize_masks.py \
  --annotation-json data/chexlocalize/gt_segmentations_val.json \
  --source-image-dir data/chexlocalize/images \
  --output-dir data/xray_lesion_seg \
  --image-size 256
```

The output layout is:

```text
data/xray_lesion_seg/
  images/
    case001.png
  masks/
    case001.png
```

By default, the script unions these CheXlocalize pathologies into a single
binary lesion mask:

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Enlarged Cardiomediastinum
- Lung Lesion
- Lung Opacity
- Pleural Effusion
- Pneumothorax
- Support Devices

If the downloaded CheXlocalize JSON uses compressed COCO RLE, install
`pycocotools` in the training environment:

```bash
pip install pycocotools
```

## 2. Train Keras U-Net

```bash
cd /Users/kimseonmin/HELIOS/new/HELIOS-Main/helios_ai

python3 preprocessing/segmentation/train_xray_lesion_unet.py \
  --image-dir data/xray_lesion_seg/images \
  --mask-dir data/xray_lesion_seg/masks \
  --output-dir preprocessing/segmentation/xray_lesion_saved_model \
  --image-size 256 \
  --batch-size 8 \
  --epochs 20
```

## 3. Convert To TF.js

```bash
cd /Users/kimseonmin/HELIOS/new/HELIOS-Main/helios_ai

preprocessing/segmentation/convert_xray_lesion_to_tfjs.sh \
  preprocessing/segmentation/xray_lesion_saved_model \
  ../Heliosclient/public/models/xray_lesion_seg_tfjs
```

The browser auto-labeler loads:

```text
Heliosclient/public/models/xray_lesion_seg_tfjs/model.json
```

No patient image is uploaded to `helios_ai` or an external AI API during
browser auto-labeling.
