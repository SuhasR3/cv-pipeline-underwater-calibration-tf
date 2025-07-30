#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Deepcalib++ Project Managers
"""
# deepcalib/models/resnet18_dual_hybrid.py
#
# Dual‑stream ResNet‑18 for DeepCalib++:
#   • stream‑1: RGB  (3‑ch)
#   • stream‑2: Sobel edge map (1‑ch)
#   • optional weight sharing between the two backbones
#   • hybrid head (cls_f, cls_xi, reg) — identical keys
#
# This makes the network plug‑compatible with
#   models/losses.py  and  the existing classification training script.
from __future__ import annotations
from typing import Dict
import tensorflow as tf
from tensorflow.keras import layers, Model

# ------------------------------------------------------------
# Re‑usable backbone wrapper (ResNet‑18 up to GAP)
# ------------------------------------------------------------
class _ResNet18Backbone(tf.keras.layers.Layer):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Start with a base ResNet18
        inputs = layers.Input(shape=(299, 299, in_channels))
        x = inputs
        
        # First conv layer - custom for 1-channel if needed
        if in_channels == 1:
            x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(x)
        else:
            x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(x)
            
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        
        # ResNet blocks (simplified representation)
        # Block 1
        x = self._make_basic_block(x, 64, 2)
        # Block 2
        x = self._make_basic_block(x, 128, 2, downsample=True)
        # Block 3
        x = self._make_basic_block(x, 256, 2, downsample=True)
        # Block 4
        x = self._make_basic_block(x, 512, 2, downsample=True)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        self.model = Model(inputs, x)
        self.out_dim = 512  # Same as PyTorch version
        
    def _make_basic_block(self, x, filters, blocks, downsample=False):
        if downsample:
            # Downsampling shortcut
            shortcut = layers.Conv2D(filters, kernel_size=1, strides=2, padding='same', use_bias=False)(x)
            shortcut = layers.BatchNormalization()(shortcut)
            x = self._basic_block(x, filters, stride=2, shortcut=shortcut)
        else:
            # First block without downsampling
            x = self._basic_block(x, filters, stride=1)
            
        # Remaining blocks
        for _ in range(1, blocks):
            x = self._basic_block(x, filters)
            
        return x
    
    def _basic_block(self, x, filters, stride=1, shortcut=None):
        if shortcut is None and (stride != 1 or x.shape[-1] != filters):
            # Create shortcut connection if dimensions change
            shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(x)
            shortcut = layers.BatchNormalization()(shortcut)
        elif shortcut is None:
            shortcut = x
            
        # First conv
        y = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        
        # Second conv
        y = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        
        # Add shortcut
        y = layers.add([y, shortcut])
        y = layers.ReLU()(y)
        
        return y
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        # [B,299,299,C] → [B,512]
        return self.model(x)

# ------------------------------------------------------------
# Hybrid coarse‑to‑fine head (same as single‑stream module)
# ------------------------------------------------------------
class HybridHead(tf.keras.layers.Layer):
    def __init__(self, in_dim: int, f_bins: int, xi_bins: int,
                 hidden: int = 256, dropout_p: float = 0.0):
        super().__init__()
        layers_list = [layers.Dense(hidden), layers.ReLU()]
        if dropout_p: layers_list.append(layers.Dropout(dropout_p))
        self.reduction = tf.keras.Sequential(layers_list)
        self.cls_f  = layers.Dense(f_bins)
        self.cls_xi = layers.Dense(xi_bins)
        self.delta  = layers.Dense(2)
        
    def call(self, z: tf.Tensor) -> Dict[str, tf.Tensor]:
        z = self.reduction(z)
        return {
            "cls_f":  self.cls_f(z),
            "cls_xi": self.cls_xi(z),
            "reg":    self.delta(z),
        }

# ------------------------------------------------------------
# Dual‑stream network
# ------------------------------------------------------------
class ResNet18DualHybrid(tf.keras.Model):
    """
    Parameters
    ----------
    share_weights : bool
        If True, both RGB and edge streams reuse *one* backbone instance;
        if False, each stream gets its own ResNet‑18 copy.
    """
    def __init__(
        self,
        f_bins: int = 46,
        xi_bins: int = 61,
        hidden: int = 256,
        dropout_p: float = 0.0,
        share_weights: bool = False,
    ):
        super().__init__()
        # 1.  Backbones -----------------------------------------------------
        self.rgb_backbone = _ResNet18Backbone(in_channels=3)
        if share_weights:
            self.edge_backbone = self.rgb_backbone              # pointer alias
        else:
            self.edge_backbone = _ResNet18Backbone(in_channels=1)
            
        # 2.  Fusion + head -------------------------------------------------
        fusion_dim = self.rgb_backbone.out_dim + self.edge_backbone.out_dim  # 1024
        self.fuse = tf.keras.Sequential([
            layers.Dense(hidden),
            layers.ReLU()
        ])
        self.head = HybridHead(hidden, f_bins, xi_bins,
                               hidden=hidden, dropout_p=dropout_p)
                               
    def call(self, rgb: tf.Tensor, edge: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        rgb  : [B, 299, 299, 3]  (‑1 … 1)
        edge : [B, 299, 299, 1]  (‑1 … 1)  – Sobel map
        """
        rgb_feat  = self.rgb_backbone(rgb)      # [B,512]
        edge_feat = self.edge_backbone(edge)    # [B,512] (or shared)
        fused     = self.fuse(tf.concat([rgb_feat, edge_feat], axis=1))
        return self.head(fused)