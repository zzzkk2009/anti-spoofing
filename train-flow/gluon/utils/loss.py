# coding: utf-8
# pylint: disable=arguments-differ
""" losses for training neural networks """
from __future__ import absolute_import
__all__ = ['Loss', 'L2Loss', 'L1Loss',
           'SigmoidBinaryCrossEntropyLoss', 'SigmoidBCELoss',
           'SoftmaxCrossEntropyLoss', 'SoftmaxCELoss',
           'KLDivLoss', 'CTCLoss', 'HuberLoss', 'HingeLoss',
           'SquaredHingeLoss', 'LogisticLoss', 'TripletLoss', 'PoissonNLLLoss', 'CosineEmbeddingLoss']

import numpy as np
from mxnet import ndarray
from .base import numeric_types
from mxnet.gluon import HybridBlock

def _apply_weighting(F, loss, weight=None, sample_weight=None):
    """Apply weighting to loss.
    Parameters
    ----------
    loss : Symbol
        The loss to be weighted.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch separately, `sample_weight` should have
        shape (64, 1).
    Returns
    -------
    loss : Symbol
        Weighted loss
    """
    if sample_weight is not None:
        loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight

    return loss

def _reshape_like(F, x, y):
    """Reshapes x to the same shape as y."""
    return x.reshape(y.shape) if F is ndarray else F.reshape_like(x, y)

class Loss(HybridBlock):
    """Base class for loss.
    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight, batch_axis, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self._weight = weight
        self._batch_axis = batch_axis

    def __repr__(self):
        s = '{name}(batch_axis={_batch_axis}, w={_weight})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Overrides to construct symbolic graph for this `Block`.
        Parameters
        ----------
        x : Symbol or NDArray
            The first input tensor.
        *args : list of Symbol or list of NDArray
            Additional input tensors.
        """
        # pylint: disable= invalid-name
        raise NotImplementedError

class SoftmaxCrossEntropyLoss(Loss):
    r"""注释
    """
    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        # Here `F` can be either mx.nd or mx.sym
        if not self._from_logits:
            pred = F.log_softmax(pred, self._axis)
        if self._sparse_label:
            # Picks elements from an input array according to the input indices along the given axis.
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred*label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)