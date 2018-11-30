import mxnet as mx
import logging
import numpy as np
import argparse
from ShuffleNet import get_shufflenet

# logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)

#数据路径
train_data = np.concatenate((mnist['train_data'], mnist['train_data'], mnist['train_data']), 
	                        axis=1)
val_data = np.concatenate((mnist['test_data'], mnist['test_data'], mnist['test_data']), 
	                       axis=1)

train_iter = mx.io.NDArrayIter(train_data, mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(val_data, mnist['test_label'], batch_size)
batch_size = 128

shufflenet = get_shufflenet()

shufflenet_mod = mx.mod.Module(symbol=shufflenet, 
                        context=[mx.gpu(0), mx.gpu(1)],
                        data_names=['data'],
                        label_names=['softmax_label'])

shufflenet_mod.fit(train_iter, 
              eval_data=val_iter, 
              optimizer='sgd',  
              optimizer_params={'learning_rate':0.01},  
              eval_metric='acc',  
              #batch_end_callback = mx.callback.Speedometer(batch_size, 20), 
              num_epoch=10) 

# parse args
parser = argparse.ArgumentParser(description="train cifar10",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
fit.add_fit_args(parser)
data.add_data_args(parser)
data.add_data_aug_args(parser)
data.set_data_aug_level(parser, 2)
parser.set_defaults(
    # network
    network        = 'resnet',
    num_layers     = 110,
    # data
    data_train     = train_fname,
    data_val       = val_fname,
    num_classes    = 10,
    num_examples  = 50000,
    image_shape    = '3,28,28',
    pad_size       = 4,
    # train
    batch_size     = 128,
    num_epochs     = 300,
    lr             = .05,
    lr_step_epochs = '200,250',
)
args = parser.parse_args()

# load network
from importlib import import_module
net = import_module('symbols.'+args.network)
sym = net.get_symbol(**vars(args))

# train
fit.fit(args, sym, data.get_rec_iter)
