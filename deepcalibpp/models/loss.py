#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepcalibpp/models/loss.py
---------------------
Hybrid "classification + residual" loss used by DeepCalib++.
The network is expected to output a dictionary
    preds = {
        'cls_f' : Tensor[B,  F_BINS],     # coarse focal‑length logits
        'cls_xi': Tensor[B, XI_BINS],     # coarse distortion  logits
        'reg'   : Tensor[B, 2]            # residual (Δf, Δξ)
    }
Ground‑truth must be supplied as
    target = {
        'f_bin' : Tensor[B],              # integer bin index  (0 … F_BINS‑1)
        'xi_bin': Tensor[B],              # integer bin index  (0 … XI_BINS‑1)
        'delta' : Tensor[B, 2]            # residual ∈ (‑0.5 … 0.5)
    }
Residuals are expressed as *fraction of the bin width*, exactly as in the
original DeepCalib paper (0 means bin centre, −0.5 means left edge, …).
The total loss is
    L = λ_cls · (CE_f + CE_xi)  +  λ_reg · MSE(Δ)
By default λ_cls = 1, λ_reg = 10 (the values the paper used).
"""
from __future__ import annotations
import tensorflow as tf

def hybrid_loss(
    preds: dict[str, tf.Tensor],
    target: dict[str, tf.Tensor],
    lambda_cls: float = 1.0,
    lambda_reg: float = 10.0,
    reduction: str = "mean",
) -> tf.Tensor:
    """
    Parameters
    ----------
    preds : dict
        Output of the network (see docstring above).
    target : dict
        Ground‑truth dictionary (see docstring above).
    lambda_cls : float
        Weight for the classification loss terms.
    lambda_reg : float
        Weight for the residual regression term.
    reduction : {'mean', 'sum', 'none'}
        Same semantics as PyTorch losses.
    Returns
    -------
    tf.Tensor
        The total hybrid loss (scalar unless reduction='none').
    """
    # 1)  Coarse classification losses (cross‑entropy)
    loss_cls_f = tf.keras.losses.sparse_categorical_crossentropy(
        target["f_bin"], preds["cls_f"], from_logits=True
    )
    loss_cls_xi = tf.keras.losses.sparse_categorical_crossentropy(
        target["xi_bin"], preds["cls_xi"], from_logits=True
    )
    
    # 2)  Fine residual regression loss (mean‑squared‑error)
    residuals = target["delta"] - preds["reg"]
    loss_reg = tf.reduce_mean(tf.square(residuals), axis=1)
    
    # Apply reduction
    if reduction == "mean":
        loss_cls_f  = tf.reduce_mean(loss_cls_f)
        loss_cls_xi = tf.reduce_mean(loss_cls_xi)
        loss_reg    = tf.reduce_mean(loss_reg)
    elif reduction == "sum":
        loss_cls_f  = tf.reduce_sum(loss_cls_f)
        loss_cls_xi = tf.reduce_sum(loss_cls_xi)
        loss_reg    = tf.reduce_sum(loss_reg)
    # For 'none', no reduction is needed
    
    # 3)  Weighted sum
    loss = lambda_cls * (loss_cls_f + loss_cls_xi) + lambda_reg * loss_reg
    return loss
