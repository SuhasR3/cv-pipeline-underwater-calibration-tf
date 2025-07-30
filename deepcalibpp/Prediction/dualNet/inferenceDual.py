#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python
"""
Inference for Dual‑Stream ResNet‑18 Hybrid model.
Produces predictions_dual.csv with columns: filename,f_pred,xi_pred
"""

import argparse, glob, numpy as np, pandas as pd, torch
from pathlib import Path
from torchvision import transforms
from torchvision.io import read_image

from deepcalibpp.models.resnet18_dual_hybrid import ResNet18DualHybrid

def decode(preds, classes_f, classes_xi):
    fb = preds['cls_f'].argmax(1).item()
    xib= preds['cls_xi'].argmax(1).item()
    df,dx= preds['reg'].squeeze(0).tolist()
    cf, cxi = float(classes_f[fb]), float(classes_xi[xib])
    wf = classes_f[1]-classes_f[0]
    wxi=classes_xi[1]-classes_xi[0]
    return cf + df*wf, cxi + dx*wxi

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',      required=True)
    p.add_argument('--data_root', default='datasets/test')
    p.add_argument('--edge_root', default='datasets/edge_sobel/test')
    p.add_argument('--share',     type=int, choices=[0,1], default=1)
    p.add_argument('--output',    default='predictions_dual.csv')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes_f  = np.arange(50, 501, 10)
    classes_xi = np.arange(0, 1.201, 0.02)

    model = ResNet18DualHybrid(len(classes_f), len(classes_xi),
                                share_weights=bool(args.share)).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    state = ckpt.get('model_state', ckpt.get('state'))
    model.load_state_dict(state); model.eval()

    tf_rgb  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299,299)),
        transforms.Normalize(mean=[0.5]*3,std=[0.5]*3),
    ])
    tf_edge = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299,299)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    rows = []
    for rgb_path in sorted(glob.glob(f'{args.data_root}/*.jpg')):
        name = Path(rgb_path).name
        rgb  = tf_rgb(read_image(rgb_path)[[2,1,0]]).unsqueeze(0).to(device)
        edge = tf_edge(read_image(str(Path(args.edge_root)/name))[0:1])\
                   .unsqueeze(0).to(device)

        with torch.no_grad(): preds = model(rgb, edge)
        f_pred, xi_pred = decode(preds, classes_f, classes_xi)
        rows.append((name, f_pred, xi_pred))

    pd.DataFrame(rows, columns=['filename','f_pred','xi_pred'])\
      .to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} to {args.output}")

if __name__=='__main__':
    main()
