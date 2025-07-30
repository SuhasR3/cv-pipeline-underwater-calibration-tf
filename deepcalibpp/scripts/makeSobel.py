#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DeepCalib++ Project Managers
"""

#!/usr/bin/env python
"""
Precompute 1‑channel Sobel edge maps for all splits.
Writes to datasets/edge_sobel/{train,valid,test}/ with the same filenames.
"""

import glob
import cv2
from pathlib import Path

SRC_ROOT  = Path('DeepCalibV2/dataset/dataset_continuous')     # RGB images live here
EDGE_ROOT = Path('DeepCalibV2/deepcalibpp/sobelData')    # output folder

for split in ('test','train','valid'):
    out_dir = EDGE_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for rgb_path in glob.glob(str(SRC_ROOT / split / '*.jpg')):
        fname = Path(rgb_path).name
        img   = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        # 1st‑order Sobel in x and y, then magnitude
        sx    = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
        sy    = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
        sobel = cv2.convertScaleAbs(cv2.magnitude(sx.astype(float), sy.astype(float)))
        cv2.imwrite(str(out_dir / fname), sobel)
    print(f"Precomputed Sobel for {split}: {len(glob.glob(str(out_dir/'*.jpg')))} files")
