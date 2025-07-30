"""
@author: DeepCalib++ Project Managers
"""
# deepcalib/models/resnet18_single_hybrid.py
#
# Author: DeepCalib++  ·  May‑2025
#
# One–stream ResNet‑18 backbone that predicts focal length (f) and radial
# distortion (ξ) from a 299×299 RGB image, using *both* coarse‑bin
# classification and fine residual regression – exactly the training
# paradigm used in the original DeepCalib paper's "Classification" branch.

from __future__ import annotations
from typing import Dict

import tensorflow as tf
from tensorflow.keras import layers

# Import the custom ResNet18 implementation
# from .CustomResNet18 import ResNet18
try:
    from .CustomResNet18 import ResNet18
    print("Successfully imported custom ResNet18")
except ImportError as e:
    print(f"Error importing CustomResNet18: {e}")
    # Fall back to a different approach
    from tensorflow.keras.applications import ResNet50
    print("Falling back to ResNet50")

# -------------------------------------------------------------
# Utility: coarse‑to‑fine hybrid head
# -------------------------------------------------------------
class HybridHead(tf.keras.layers.Layer):
    """
    Parameters
    ----------
    in_dim : int
        Dimensionality of the flattened CNN feature vector.
    f_bins : int
        Number of classification bins for focal length.
    xi_bins : int
        Number of classification bins for distortion ξ.
    hidden : int
        Width of the shared hidden layer before the three output heads.
    dropout_p : float
        Optional dropout for regularisation (0 disables).
    """
    def __init__(
        self,
        in_dim: int = 512,
        f_bins: int = 46,
        xi_bins: int = 61,
        hidden: int = 256,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        
        layers_list = [
            layers.InputLayer(input_shape=(in_dim,)),
            layers.Dense(hidden),
            layers.ReLU()
        ]
        #layers_list = [layers.Dense(hidden, input_dim=in_dim), layers.ReLU()]
        #layers_list = [layers.Dense(hidden,layers.ReLU())]
        if dropout_p > 0:
            layers_list.append(layers.Dropout(dropout_p))
        self.reduction = tf.keras.Sequential(layers_list)

        # Two soft‑max classification heads (coarse bins)
        self.cls_f  = layers.Dense(f_bins)
        self.cls_xi = layers.Dense(xi_bins)

        # Two‑value residual regression inside chosen bin
        self.delta  = layers.Dense(2)          # (Δf, Δξ)

    def call(self, z: tf.Tensor) -> Dict[str, tf.Tensor]:
        z = self.reduction(z)
        return {
            "cls_f":  self.cls_f(z),                # [B, f_bins]
            "cls_xi": self.cls_xi(z),               # [B, xi_bins]
            "reg":    self.delta(z)                 # [B, 2]
        }


# -------------------------------------------------------------
# Single‑stream ResNet‑18 backbone + HybridHead
# -------------------------------------------------------------
class ResNet18SingleHybrid(tf.keras.Model):
    """
    One‑stream (RGB‑only) network that matches DeepCalib's single‑net
    architecture but uses ResNet‑18 features and the HybridHead above.

    Input  : RGB tensor [B, 299, 299, 3] in ‑1…1 range
    Output : dict{ 'cls_f', 'cls_xi', 'reg' }  — ready for DeepCalib loss
    """
    def __init__(
        self,
        f_bins: int = 46,
        xi_bins: int = 61,
        hidden: int = 256,
        dropout_p: float = 0.0,
        pretrained: bool = False,
    ):
        super().__init__()

        # 1) Backbone -------------------------------------------------------
        weights = None
        if pretrained:
            weights = "imagenet"
        
        # Use our custom ResNet18 implementation instead of ResNet50
        backbone = ResNet18(
            input_shape=(299, 299, 3),
            include_top=False,  # We don't need the classification head
            weights=weights
        )
        self.features = backbone

        # 2) Hybrid head ----------------------------------------------------
        self.head = HybridHead(
            in_dim=512,  # ResNet18 produces 512 features
            f_bins=f_bins,
            xi_bins=xi_bins,
            hidden=hidden,
            dropout_p=dropout_p,
        )

    def call(self, rgb: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Parameters
        ----------
        rgb : tf.Tensor
            Shape [B, 299, 299, 3] — image pre‑processed to (‑1,1).

        Returns
        -------
        Dict[str, tf.Tensor]
            'cls_f'  : coarse focal‑length logits   [B, f_bins]
            'cls_xi' : coarse distortion logits    [B, xi_bins]
            'reg'    : residual Δf, Δξ             [B, 2]
        """
        # ResNet‑18 backbone ⇒ [B, 512]
        feat = self.features(rgb)
        feat = tf.keras.layers.GlobalAveragePooling2D()(feat)
        return self.head(feat)                     # dict output