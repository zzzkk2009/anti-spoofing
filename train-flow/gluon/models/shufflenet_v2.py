# -*- coding:utf-8 -*-

import os
import mxnet as mx
from mxnet.gluon import nn

def _add_conv(out, channels=1, kernel=1, stride=1, pad=0, num_group=1, activation=True):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True))
    if activation:
        out.add(nn.Activation('relu'))

def _add_conv_1x1(out, channels=1):
    out.add(nn.Conv2D(channels, 1, 1, 0, use_bias=False))
    out.add(nn.BatchNorm(scale=True))
    out.add(nn.Activation('relu'))

class ShuffleChannels(nn.HybridBlock):
    def __init__(self, groups=3, **kwargs):
        super(ShuffleChannels, self).__init__()
        self.groups = groups

    def hybrid_forward(self, F, x, *args, **kwargs):
        data = F.reshape(x, shape=(0, -4, self.groups, -1, -2))
        data = F.swapaxes(data, 1, 2)
        data = F.reshape(data, shape=(0, -3, -2))
        return data

class InvertedResidual(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, stride, benchmodel=1, stage=0):
        super().__init__()
        self.stride = stride
        self.benchmodel = benchmodel
        self.shuffle_channels = ShuffleChannels(2)
        assert stride in [1, 2]

        split_channel = out_channels // 2

        if benchmodel == 1:
            self.branch2 = nn.HybridSequential(prefix=f'{stage}-branch2_') # f-string python>=3.6语法(formated string)
            # pw
            _add_conv(self.branch2, split_channel, 1, 1, 0)
            # dw
            _add_conv(self.branch2, split_channel, 3, stride, 1, num_group=split_channel, activation=False)
            # pw
            _add_conv(self.branch2, split_channel, 1, 1, 0)
        else:
            self.branch1 = nn.HybridSequential(prefix=f'{stage}-branch1_')
            # pw
            self.branch1.add(nn.Conv2D(in_channels, 3, stride, 1, groups=in_channels, use_bias=False))
            # pw
            self.branch1.add(nn.Conv2D(split_channel, 1, 1, use_bias=False))
            self.branch1.add(nn.Activation('relu'))

            self.branch2 = nn.HybridSequential(prefix=f'{stage}-branch2_')
            # pw
            _add_conv(self.branch2, split_channel, 1, 1, 0)
            # dw
            _add_conv(self.branch2, split_channel, 3, stride, 1, num_group=split_channel, activation=False)
            # pw
            _add_conv(self.branch2, split_channel, 1, 1, 0)

    def hybrid_forward(self, F, x, *args, **kwargs):
        if 1 == self.benchmodel:
            x1, x2 = F.split(x, num_outputs=2, axis=1)
            out = F.concat(x1, self.branch2(x2), dim=1)
        elif 2 == self.benchmodel:
            out = F.concat(self.branch1(x), self.branch2(x), dim=1)
        else:
            out = x

        return self.shuffle_channels(out)


class ShuffleNetV2(nn.HybridBlock):
    def __init__(self, num_classes, width_multiplier):
        super().__init__()
        width_config={
            0.5: (24, 48, 96, 192, 1024),
            1.0: (24, 116, 232, 464, 1024),
            1.5: (24, 176, 352, 704, 1024),
            2.0: (24, 244, 488, 976, 2048)
        }
        channel_config = width_config[width_multiplier]
        self.num_classes = num_classes
        self.stage_repeats = [4, 8, 4]

        in_channel = channel_config[0]

        self.features = nn.HybridSequential(prefix='feature_')
        with self.features.name_scope():
            _add_conv(self.features, channel_config[0], 3, stride=2, pad=1)
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

            for idx in range(len(self.stage_repeats)):
                num_repeat  = self.stage_repeats[idx]
                out_channel = channel_config[idx + 1]
                for i in range(num_repeat):
                    if i == 0:
                        self.features.add(InvertedResidual(in_channel, out_channel, 2, 2, stage=idx))
                    else:
                        self.features.add(InvertedResidual(in_channel, out_channel, 1, 1, stage=idx))
                    in_channel = out_channel

            _add_conv_1x1(self.features, channel_config[-1])
            self.features.add(nn.GlobalAvgPool2D())

        self.output = nn.HybridSequential(prefix='output_')
        self.output.add(nn.Dense(self.num_classes))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    """Test"""
    data = mx.ndarray.zeros((2, 3, 224, 224))

    net = ShuffleNetV2(num_classes=2, width_multiplier=0.5)
    net.initialize(mx.init.Xavier())
    net(data)

    # print and visualize networks structure
    print(net)
    # mx.viz.plot_network(net(mx.sym.var("data"))).view()

    # print feature map
    print('\n-----------------------------')
    print('shape of feature maps:')
    x = data
    for layer in net.features:
        x = layer(x)
        print('{:25}\t{}'.format(layer.name, x.shape))