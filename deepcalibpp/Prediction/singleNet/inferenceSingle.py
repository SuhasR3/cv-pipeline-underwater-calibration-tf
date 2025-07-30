#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python
#!/usr/bin/env python
"""
Inference for Single‑Stream ResNet‑18 Hybrid model.
Produces predictions_single.csv with columns: filename,f_pred,xi_pred
"""

import argparse, glob, numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from deepcalibpp.models.architectures.resnetSingle import ResNet18SingleHybrid


def decode(preds: dict[str, tf.Tensor], classes_f: np.ndarray, classes_xi: np.ndarray):
    fb = int(tf.argmax(preds['cls_f'], axis=1).numpy()[0])
    xib = int(tf.argmax(preds['cls_xi'], axis=1).numpy()[0])
    df, dxi = preds['reg'].numpy().squeeze(0).tolist()
    cf, cxi = float(classes_f[fb]), float(classes_xi[xib])
    wf = classes_f[1] - classes_f[0]
    wxi = classes_xi[1] - classes_xi[0]
    return cf + df*wf, cxi + dxi*wxi


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',      required=True, help='Path to TF weights file (.h5)')
    p.add_argument('--data_root', default='datasets/test', help='Directory with test .jpg images')
    p.add_argument('--output',    default='predictions_single.csv', help='CSV output path')
    args = p.parse_args()

    # Device setup: TF will use GPU if available
    print("TF built with GPU support:", tf.test.is_built_with_gpu_support())
    print("Physical GPUs:", tf.config.list_physical_devices('GPU'))

    # Define bins
    classes_f  = np.arange(50, 501, 10)
    classes_xi = np.arange(0, 1.201, 0.02)

    # Build model and load weights
    model = ResNet18SingleHybrid(f_bins=len(classes_f), xi_bins=len(classes_xi))
    # Create model by calling it once
    dummy = tf.zeros((1, 299, 299, 3))
    _ = model(dummy)
    model.load_weights(args.ckpt)
    model.trainable = False

    rows = []
    for img_path in sorted(glob.glob(f'{args.data_root}/*.jpg')):
        name = Path(img_path).name
        # Load image and preprocess
        img = load_img(img_path, target_size=(299, 299))
        arr = img_to_array(img)[:, :, ::-1]  # BGR→RGB
        inp = (arr / 127.5 - 1.0)[None, ...]  # scale to [-1,1] and add batch dim
        inp_tensor = tf.convert_to_tensor(inp, dtype=tf.float32)
        # Inference
        preds = model(inp_tensor, training=False)
        f_pred, xi_pred = decode(preds, classes_f, classes_xi)
        rows.append((name, f_pred, xi_pred))

    # Save results
    pd.DataFrame(rows, columns=['filename','f_pred','xi_pred']).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == '__main__':
    main()