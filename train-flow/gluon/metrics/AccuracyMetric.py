# -*-coding:utf-8-*-

from __future__ import absolute_import
import math
import numpy

from mxnet import ndarray
from ../ import registry

def check_label_shapes(labels, preds, wrap=False, shape=False):
    """Helper function for checking shape of label and prediction

    Parameters
    ----------
    labels : list of `NDArray`
        The labels of the data.

    preds : list of `NDArray`
        Predicted values.

    wrap : boolean
        If True, wrap labels/preds in a list if they are single NDArray

    shape : boolean
        If True, check the shape of labels and preds;
        Otherwise only check their length.
    """
    if not shape:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

    if wrap:
        if isinstance(labels, ndarray.ndarray.NDArray):
            labels = [labels]
        if isinstance(preds, ndarray.ndarray.NDArray):
            preds = [preds]

    return labels, preds

@register
@alias('acc')
class Accuracy(mx.metric.EvalMetric):
    """Computes accuracy classification score.

    The accuracy score is defined as

    .. math::
        \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1}
        \\text{1}(\\hat{y_i} == y_i)

    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> acc = mx.metric.Accuracy()
    >>> acc.update(preds = predicts, labels = labels)
    >>> print acc.get()
    ('accuracy', 0.6666666666666666)
    """
    def __init__(self, axis=1, name='accuracy',
                 output_names=None, label_names=None):
        super(Accuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names,
            has_global_stats=True)
        self.axis = axis

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data with class indices as values, one per sample.

        preds : list of `NDArray`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')
            # flatten before checking shapes to avoid shape miss match
            label = label.flat
            pred_label = pred_label.flat

            check_label_shapes(label, pred_label)

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += len(pred_label)
            self.global_num_inst += len(pred_label)