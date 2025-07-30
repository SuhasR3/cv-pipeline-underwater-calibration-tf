#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Deepcalib++ Project Managers
"""
#!/usr/bin/env python
# network_training/Classification/Dual_Net/train_resnet18_dual_hybrid.py
# ----------------------------------------------------------------------
# DeepCalib++ · Dual‑stream  ·  RGB  +  Sobel edge
#   • share --share 1   → RGB & Edge use SAME backbone weights
#   • share --share 0   → independent backbones (more capacity)
#
# Output heads:  cls_f , cls_xi  (coarse)  +  reg (Δf, Δξ)
# Loss        :  hybrid_loss  =  CE_f + CE_xi + λ·MSE(Δ)      (λ=10)
#
# Stops early when val_loss fails to improve for --patience epochs.
# ------------------------------------------------------------------
# DeepCalib++ · Dual‑Net ResNet‑18 Hybrid Training w/ Accuracy Logging 
# ------------------------------------------------------------------
from __future__ import annotations
import random, glob, datetime, argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# TensorBoard
from keras.callbacks import TensorBoard
from keras.backend.tensorflow_backend import set_session
from deepcalibpp.models.resnet18_dual_hybrid_tf import ResNet18DualHybrid
from deepcalibpp.losses import hybrid_loss
# ---------------------------------------------------------------------------- #
# GPU config
# ---------------------------------------------------------------------------- #
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='datasets', help='RGB train/valid/')
parser.add_argument('--edge_root', default='datasets/edge_sobel', help='Sobel train/valid/')
parser.add_argument('--share',     type=int, choices=[0,1], default=1)
parser.add_argument('--batch',     type=int, default=48)
parser.add_argument('--epochs',    type=int, default=200)
parser.add_argument('--patience',  type=int, default=10)
parser.add_argument('--lr',        type=float, default=1e-3)
parser.add_argument('--save_root', default='new_logs')
args = parser.parse_args()
# ---------------------------------------------------------------------------- #
# Bins
# ---------------------------------------------------------------------------- #
classes_focal      = np.arange(50, 501, 10)
classes_distortion = np.arange(0, 1.201, 0.02)
F_BINS, XI_BINS    = len(classes_focal), len(classes_distortion)
# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #
class DualStreamDataset(Sequence):
    def __init__(self, files, batch_size=32, shuffle=True):
        self.files = files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(files))
        if shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self): 
        return int(np.ceil(len(self.files) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_rgb = []
        batch_edge = []
        batch_f_bin = []
        batch_xi_bin = []
        
        for i in batch_indices:
            rgb_path = self.files[i]
            edge_path = str(Path(args.edge_root)/Path(rgb_path).name)
            
            # Load and normalize RGB image
            rgb = load_img(rgb_path, target_size=(299, 299))
            rgb = img_to_array(rgb)[:,:,::-1]  # BGR→RGB
            rgb = rgb / 127.5 - 1.0  # → [‑1,1]
            
            # Load and normalize edge image
            edge = load_img(edge_path, target_size=(299, 299), color_mode='grayscale')
            edge = img_to_array(edge)
            edge = edge / 127.5 - 1.0  # → [‑1,1]
            
            f_val = float(Path(rgb_path).stem.split('_f_')[1].split('_d_')[0])
            xi_val = float(Path(rgb_path).stem.split('_d_')[1])
            f_bin = int((f_val - classes_focal[0])/(classes_focal[1]-classes_focal[0]))
            xi_bin = int(round(xi_val/(classes_distortion[1]-classes_distortion[0])))
            
            batch_rgb.append(rgb)
            batch_edge.append(edge)
            batch_f_bin.append(f_bin)
            batch_xi_bin.append(xi_bin)
            
        batch_rgb = np.array(batch_rgb)
        batch_edge = np.array(batch_edge)
        batch_f_bin = np.array(batch_f_bin)
        batch_xi_bin = np.array(batch_xi_bin)
        batch_delta = np.zeros((len(batch_indices), 2))  # Zero deltas as in original
        
        return [batch_rgb, batch_edge], {
            'f_bin': batch_f_bin,
            'xi_bin': batch_xi_bin,
            'delta': batch_delta
        }
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def make_split(split):
    p = sorted(glob.glob(f'{args.data_root}/{split}/*.jpg'))
    random.shuffle(p)
    return p

train_loader = DualStreamDataset(make_split('train'), batch_size=args.batch, shuffle=True)
val_loader = DualStreamDataset(make_split('valid'), batch_size=args.batch, shuffle=False)
# ---------------------------------------------------------------------------- #
# Early‑Stopping
# ---------------------------------------------------------------------------- #
class EarlyStopper:
    def __init__(self, patience:int, min_delta:float=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.count = float('inf'), 0
    def step(self, loss):
        if loss < self.best - self.min_delta:
            self.best, self.count = loss, 0
        else:
            self.count += 1
        return self.count < self.patience
stopper = EarlyStopper(patience=args.patience)
# ---------------------------------------------------------------------------- #
# Model, Opt, TensorBoard
# ---------------------------------------------------------------------------- #
model = ResNet18DualHybrid(
    f_bins=F_BINS, xi_bins=XI_BINS,
    share_weights=bool(args.share)
)
model.build(input_shape=[(None, 299, 299, 3), (None, 299, 299, 1)])  # Build model
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

stamp   = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
dir_n   = f'dual_sobel_share{args.share}'
run_dir = Path(args.save_root)/stamp/dir_n
log_dir = run_dir/'Log'; best_dir = run_dir/'Best'
log_dir.mkdir(parents=True, exist_ok=True)
best_dir.mkdir(parents=True, exist_ok=True)
tensorboard = TensorBoard(log_dir=str(log_dir))
tensorboard.set_model(model)
best_val = float('inf')
# ---------------------------------------------------------------------------- #
# Training loop w/ accuracy logging
# ---------------------------------------------------------------------------- #
for epoch in range(args.epochs):
    # Training phase
    tr_ls, tr_acc_f, tr_acc_xi = [], [], []
    
    for batch_idx in range(len(train_loader)):
        [rgb, edge], targets = train_loader[batch_idx]
        f_bin, xi_bin = targets['f_bin'], targets['xi_bin']
        
        with tf.GradientTape() as tape:
            # Forward pass
            preds = model([rgb, edge], training=True)
            loss_value = hybrid_loss(targets, preds)
        
        # Compute gradients and update weights
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        # Track metrics
        tr_ls.append(loss_value.numpy())
        
        # Calculate accuracy
        f_pred = tf.argmax(preds['cls_f'], axis=1).numpy()
        xi_pred = tf.argmax(preds['cls_xi'], axis=1).numpy()
        
        f_accuracy = np.mean(f_pred == f_bin)
        xi_accuracy = np.mean(xi_pred == xi_bin)
        
        tr_acc_f.append(f_accuracy)
        tr_acc_xi.append(xi_accuracy)
    
    # Validation phase
    val_ls, val_acc_f, val_acc_xi = [], [], []
    
    for batch_idx in range(len(val_loader)):
        [rgb, edge], targets = val_loader[batch_idx]
        f_bin, xi_bin = targets['f_bin'], targets['xi_bin']
        
        # Forward pass (no training)
        preds = model([rgb, edge], training=False)
        l = hybrid_loss(targets, preds)
        
        # Track metrics
        val_ls.append(l.numpy())
        
        # Calculate accuracy
        f_pred = tf.argmax(preds['cls_f'], axis=1).numpy()
        xi_pred = tf.argmax(preds['cls_xi'], axis=1).numpy()
        
        f_accuracy = np.mean(f_pred == f_bin)
        xi_accuracy = np.mean(xi_pred == xi_bin)
        
        val_acc_f.append(f_accuracy)
        val_acc_xi.append(xi_accuracy)
    
    # epoch metrics
    t_loss, v_loss = np.mean(tr_ls), np.mean(val_ls)
    t_accf, v_accf = np.mean(tr_acc_f), np.mean(val_acc_f)
    t_accx, v_accx = np.mean(tr_acc_xi), np.mean(val_acc_xi)
    
    # TensorBoard
    logs = {
        'loss':      t_loss, 'val_loss':  v_loss,
        'acc_f':     t_accf, 'val_acc_f': v_accf,
        'acc_xi':    t_accx,'val_acc_xi': v_accx
    }
    tensorboard.on_epoch_end(epoch, logs)
    
    print(f"[E{epoch:03d}] loss {t_loss:.4f}/{v_loss:.4f}"
          f" acc_f {t_accf:.3f}/{v_accf:.3f}"
          f" acc_xi {t_accx:.3f}/{v_accx:.3f}")
    
    # checkpoint best
    if v_loss < best_val:
        best_val = v_loss
        model.save_weights(
            str(best_dir/f'weights_e{epoch:03d}_v{v_loss:.4f}.h5')
        )
    
    # early stop?
    if not stopper.step(v_loss):
        print(f"Early stopping at epoch {epoch}")
        break
    
    # LR decay
    new_lr = args.lr * (0.1 ** (epoch//2))
    optimizer.learning_rate.assign(new_lr)

tensorboard.on_train_end(None)