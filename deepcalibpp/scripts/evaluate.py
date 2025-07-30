#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/evaluate.py

Evaluate DeepCalib++ predictions:
 - MAE for focal length and distortion
 - Bin‑classification reports
 - Confusion‑matrix plots
"""

import argparse, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_csv', required=True,
                   help='CSV with columns filename,f_pred,xi_pred')
    p.add_argument('--out_dir', default='eval', help='Where to write outputs')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) load predictions
    df = pd.read_csv(args.pred_csv)

    # 2) extract ground truth from filename
    df['f_gt']  = df.filename.str.extract(r'_f_(\d+)_').astype(float)
    df['xi_gt'] = df.filename.str.extract(r'_d_([\d\.]+)\.jpg').astype(float)

    # 3) define bin centres (must match training)
    classes_f  = np.arange(50, 501, 10)    # [50,60,…,500]
    classes_xi = np.arange(0, 1.201, 0.02) # [0.00,0.02,…,1.20]

    # 4) map ground truth & predictions to bin indices
    df['f_bin_gt']  = np.digitize(df.f_gt,  classes_f,  right=False) - 1
    df['xi_bin_gt'] = np.digitize(df.xi_gt, classes_xi, right=False) - 1
    df['f_bin_pred']  = np.digitize(df.f_pred,  classes_f,  right=False) - 1
    df['xi_bin_pred'] = np.digitize(df.xi_pred, classes_xi, right=False) - 1

    # 5) MAE
    mae_f  = mean_absolute_error(df.f_gt,  df.f_pred)
    mae_xi = mean_absolute_error(df.xi_gt, df.xi_pred)
    print(f"MAE focal length : {mae_f:.2f} px")
    print(f"MAE distortion   : {mae_xi:.4f}")

    # 6) classification reports
    print("\n=== Focal‑length bin classification ===")
    print(classification_report(df.f_bin_gt, df.f_bin_pred, digits=4))
    print("=== Distortion‑bin classification ===")
    print(classification_report(df.xi_bin_gt, df.xi_bin_pred, digits=4))

    # 7) confusion matrices
    cm_f  = confusion_matrix(df.f_bin_gt,  df.f_bin_pred)
    cm_xi = confusion_matrix(df.xi_bin_gt, df.xi_bin_pred)

    # 8) plot & save
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_f, cmap='Blues', cbar=True)
    plt.title('Focal‑Length Bin Confusion')
    plt.xlabel('Predicted Bin')
    plt.ylabel('True Bin')
    plt.tight_layout()
    plt.savefig(Path(args.out_dir)/'confusion_focal.png')
    plt.close()

    plt.figure(figsize=(8,6))
    sns.heatmap(cm_xi, cmap='Greens', cbar=True)
    plt.title('Distortion Bin Confusion')
    plt.xlabel('Predicted Bin')
    plt.ylabel('True Bin')
    plt.tight_layout()
    plt.savefig(Path(args.out_dir)/'confusion_distortion.png')
    plt.close()

    print(f"\nConfusion matrices saved to `{args.out_dir}`")

if __name__ == '__main__':
    main()
