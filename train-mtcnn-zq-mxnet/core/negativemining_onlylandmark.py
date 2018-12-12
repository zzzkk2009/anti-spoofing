import mxnet as mx
import numpy as np
from config import config

class NegativeMiningOperator_OnlyLandmark(mx.operator.CustomOp):
    def __init__(self, landmark_ohem=config.LANDMARK_OHEM, landmark_ohem_ratio=config.LANDMARK_OHEM_RATIO):
        super(NegativeMiningOperator_OnlyLandmark, self).__init__()
        self.landmark_ohem = landmark_ohem
        self.landmark_ohem_ratio = landmark_ohem_ratio

    def forward(self, is_train, req, in_data, out_data, aux):
        landmark_pred = in_data[0].asnumpy() # batchsize x 10
        landmark_target = in_data[1].asnumpy() # batchsize x 10
        
        # landmark
        self.assign(out_data[0], req[0], in_data[0])
        landmark_keep = np.zeros(landmark_pred.shape[0])

        if self.landmark_ohem:
            keep_num = int(len(landmark_keep) * self.landmark_ohem_ratio)
            L1_error = np.sum(abs(landmark_pred - landmark_target), axis=1)
            keep = np.argsort(L1_error)[::-1][:keep_num]
            landmark_keep[keep] = 1
        else:
            landmark_keep += 1
        self.assign(out_data[1], req[1], mx.nd.array(landmark_keep))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        landmark_keep = out_data[0].asnumpy().reshape(-1, 1)

        landmark_grad = np.repeat(landmark_keep, 10, axis=1)

        landmark_grad /= len(np.where(landmark_keep == 1)[0])

        self.assign(in_grad[0], req[0], mx.nd.array(landmark_grad))


@mx.operator.register("negativemining_onlylandmark")
class NegativeMiningProp_OnlyLandmark(mx.operator.CustomOpProp):
    def __init__(self):
        super(NegativeMiningProp_OnlyLandmark, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['landmark_pred', 'landmark_target']

    def list_outputs(self):
        return ['landmark_out', 'landmark_keep']

    def infer_shape(self, in_shape):
        #print(in_shape)
        keep_shape = (in_shape[0][0], )
        return in_shape, [in_shape[0], keep_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return NegativeMiningOperator_OnlyLandmark()
