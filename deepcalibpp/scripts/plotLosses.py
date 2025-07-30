#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/plot_losses.py

Loads TensorBoard event files and plots:
 - training loss vs steps
 - validation loss vs steps
 - focal‑accuracy vs steps
 - distortion‑accuracy vs steps
Each saved as a separate PNG in --out_dir.
"""

import argparse, os
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_scalars(logdir: str, tag: str):
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    vals  = [e.value for e in events]
    return steps, vals

def plot_curve(steps, vals, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(steps, vals, marker='o', linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', required=True,
                   help='TensorBoard log directory (e.g. new_logs/.../model_multi_class/Log)')
    p.add_argument('--out_dir', default='eval_plots',
                   help='Where to write PNG plots')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # tags must match your tensorboard.on_epoch_end keys
    tags = [
        ('loss',      'Training Loss'),
        ('val_loss',  'Validation Loss'),
        ('acc_f',     'Training F‑acc'),
        ('val_acc_f', 'Validation F‑acc'),
        ('acc_xi',    'Training ξ‑acc'),
        ('val_acc_xi','Validation ξ‑acc'),
    ]

    for tag, title in tags:
        steps, vals = load_scalars(args.logdir, tag)
        out_png = Path(args.out_dir)/f'{tag}.png'
        plot_curve(steps, vals, title, 'Epoch', tag, out_png)
        print(f"Plotted {title} to {out_png}")

if __name__ == '__main__':
    main()
