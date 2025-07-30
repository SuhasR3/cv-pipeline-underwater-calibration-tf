#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# DeepCalib++ · Single‑Net ResNet‑18 Hybrid Training w/ Accuracy Logging
# ------------------------------------------------------------------
from __future__ import annotations
import random, glob, datetime, argparse
from pathlib import Path
import os, sys

# Keep project root path as is
sys.path.append("/home/pobumnem/DeepCalibV2")

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from deepcalibpp.models.architectures.resnetSingle import ResNet18SingleHybrid
from deepcalibpp.models.loss import hybrid_loss

# ---------------------------------------------------------------------------- #
# GPU Diagnostics & Config
# ---------------------------------------------------------------------------- #
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"Physical GPUs: {gpus}")
if gpus:
    for gpu in gpus:
        tf.config.set_memory_growth(gpu, True)
    print("Enabled memory growth on GPUs")

# ---------------------------------------------------------------------------- #
# CLI arguments
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='datasets', help='root with train/valid/')
parser.add_argument('--save_root', default='new_logs', help='where to write logs/weights')
parser.add_argument('--batch',     type=int, default=60)
parser.add_argument('--epochs',    type=int, default=10000)
parser.add_argument('--patience',  type=int, default=10)
parser.add_argument('--lr',        type=float, default=1e-3)
args = parser.parse_args()

# ---------------------------------------------------------------------------- #
# Log directory & TensorBoard writer
# ---------------------------------------------------------------------------- #
stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = Path(args.save_root) / stamp / 'Log'
log_dir.mkdir(parents=True, exist_ok=True)
summary_writer = tf.summary.create_file_writer(str(log_dir))

# ---------------------------------------------------------------------------- #
# Bin definitions (as in DeepCalib)
# ---------------------------------------------------------------------------- #
classes_focal      = np.arange(50, 501, 10)    # 46 bins
classes_distortion = np.arange(0, 1.201, 0.02) # 61 bins
F_BINS, XI_BINS    = len(classes_focal), len(classes_distortion)

# ---------------------------------------------------------------------------- #
# Dataset Sequence
# ---------------------------------------------------------------------------- #
class DeepCalibDataset(Sequence):
    def __init__(self, files, labels, batch_size=32, shuffle=True):
        if len(files) != len(labels):
            raise ValueError(f"Files/labels length mismatch: {len(files)} vs {len(labels)}")
        self.files, self.labels = files, labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(files))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        idxs = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        if not len(idxs):
            raise IndexError
        batch_x, batch_f, batch_xi = [], [], []
        for i in idxs:
            img = load_img(self.files[i], target_size=(299, 299))
            arr = img_to_array(img)[:, :, ::-1]  # BGR→RGB
            batch_x.append(arr / 127.5 - 1.0)
            f_b, xi_b = self.labels[i]
            batch_f.append(f_b)
            batch_xi.append(xi_b)
        return np.stack(batch_x), {
            'f_bin': np.array(batch_f),
            'xi_bin': np.array(batch_xi),
            'delta': np.zeros((len(idxs), 2))
        }

# ---------------------------------------------------------------------------- #
# Train/Val split helper
# ---------------------------------------------------------------------------- #
def make_split(split: str):
    pattern = f'{args.data_root}/{split}/*.jpg'
    paths = sorted(glob.glob(pattern))
    labels = []
    for p in paths:
        try:
            f_val  = float(p.split('_f_')[1].split('_d_')[0])
            xi_val = float(p.split('_d_')[1].split('.jpg')[0])
            f_idx  = int((f_val - classes_focal[0]) / (classes_focal[1] - classes_focal[0]))
            xi_idx = int(round(xi_val / (classes_distortion[1] - classes_distortion[0])))
            labels.append([f_idx, xi_idx])
        except Exception:
            continue
    c = list(zip(paths, labels)); random.shuffle(c)
    if not c:
        return [], []
    pths, lbls = zip(*c)
    return list(pths), list(lbls)

train_files, train_lbl = make_split('train')
val_files,   val_lbl   = make_split('valid')
if not train_files or not val_files:
    print("No data found. Exiting."); sys.exit(1)

train_ds = DeepCalibDataset(train_files, train_lbl, batch_size=args.batch)
val_ds   = DeepCalibDataset(val_files,   val_lbl,   batch_size=args.batch, shuffle=False)

# ---------------------------------------------------------------------------- #
# EarlyStopping helper
# ---------------------------------------------------------------------------- #
class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.counter = float('inf'), 0
    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best, self.counter = val_loss, 0
        else:
            self.counter += 1
        return self.counter < self.patience

stopper = EarlyStopper(args.patience)

# ---------------------------------------------------------------------------- #
# Model, Optimizer
# ---------------------------------------------------------------------------- #
model = ResNet18SingleHybrid(f_bins=F_BINS, xi_bins=XI_BINS)
try:
    print(f"Model feature size: {model.feature_size}")
except AttributeError:
    print("WARNING: HybridHead input dim mismatch. Update ResNet18SingleHybrid.")
_ = model(tf.zeros((1, 299, 299, 3)))
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

# ---------------------------------------------------------------------------- #
# Training Loop
# ---------------------------------------------------------------------------- #
best_val = float('inf')
for epoch in range(1, args.epochs + 1):
    print(f"Epoch {epoch}/{args.epochs}")

    # Training phase
    tlosses = []
    for batch_idx in range(len(train_ds)):
        x_batch, y_batch = train_ds[batch_idx]
        with tf.GradientTape() as tape:
            preds = model(x_batch, training=True)
            loss = hybrid_loss(preds, y_batch)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        tlosses.append(loss.numpy())

    # Validation phase
    vlosses = []
    for batch_idx in range(len(val_ds)):
        x_batch, y_batch = val_ds[batch_idx]
        preds = model(x_batch, training=False)
        vlosses.append(hybrid_loss(preds, y_batch).numpy())

    t_loss, v_loss = np.mean(tlosses), np.mean(vlosses)
    print(f"  loss {t_loss:.4f}  val_loss {v_loss:.4f}")

    # Write summaries
    with summary_writer.as_default():
        tf.summary.scalar('train_loss', t_loss, step=epoch)
        tf.summary.scalar('val_loss', v_loss, step=epoch)
        tf.summary.scalar('learning_rate', optimizer.learning_rate, step=epoch)

    # Checkpoint best model
    if v_loss < best_val:
        best_val = v_loss
        out_dir = Path(args.save_root) / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(out_dir / f'best_val_{v_loss:.4f}weights.h5'))

    # Early stopping
    if not stopper.step(v_loss):
        print(f"Early stopping at epoch {epoch}")
        break

    # Step-decay LR every 2 epochs based on current LR
    if epoch % 2 == 0:
        old_lr = float(optimizer.learning_rate)
        new_lr = old_lr * 0.1
        optimizer.learning_rate.assign(new_lr)
        print(f"  ↳ reduced LR to {new_lr:.2e}")
