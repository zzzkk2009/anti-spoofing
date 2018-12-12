import mxnet as mx
import numpy as np
from config import config


class LANDMARK_MSE(mx.metric.EvalMetric):
    def __init__(self):
        super(LANDMARK_MSE, self).__init__('lmL2')

    def update(self,labels, preds):
        # output: landmark_pred_output, landmark_keep_inds
        # label: landmark_target
        pred_delta = preds[0].asnumpy()
        landmark_target = labels[0].asnumpy()

        landmark_keep = preds[1].asnumpy()
        keep = np.where(landmark_keep == 1)[0]

        pred_delta = pred_delta[keep]
        landmark_target = landmark_target[keep]

        e = (pred_delta - landmark_target)**2
        error = np.sum(e)
        self.sum_metric += error
        self.num_inst += e.size

class LANDMARK_L1(mx.metric.EvalMetric):
    def __init__(self):
        super(LANDMARK_L1, self).__init__('lmL1')

    def update(self,labels, preds):
        # output: landmark_pred_output, landmark_keep_inds
        # label: landmark_target
        pred_delta = preds[0].asnumpy()
        landmark_target = labels[0].asnumpy()

        landmark_keep = preds[1].asnumpy()
        keep = np.where(landmark_keep == 1)[0]

        pred_delta = pred_delta[keep]
        landmark_target = landmark_target[keep]

        e = abs(pred_delta - landmark_target)
        error = np.sum(e)
        self.sum_metric += error
        self.num_inst += e.size