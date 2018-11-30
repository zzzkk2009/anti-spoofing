# -*- coding:utf-8 -*-

import os
import shutil
import zipfile
import zlib
import datetime
import gluonbook as gb
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn
from ShuffleNet import get_shufflenet

transform_train = gdata.vision.transforms.Compose([
    # 将图像放大成高和宽各为 40 像素的正方形。
    #gdata.vision.transforms.Resize(40),
    # 随机对高和宽各为 40 像素的正方形图像裁剪出面积为原图像面积 0.64 到 1 倍之间的小正方
    # 形，再放缩为高和宽各为 32 像素的正方形。
    # gdata.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
    #                                           ratio=(1.0, 1.0)),
    # 随机左右翻转图像。
    gdata.vision.transforms.RandomFlipLeftRight(),
    # 将图像像素值按比例缩小到 0 和 1 之间，并将数据格式从“高 * 宽 * 通道”改为
    # “通道 * 高 * 宽”。
    gdata.vision.transforms.ToTensor(),
    # 对图像的每个通道做标准化。
    gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                      [0.2023, 0.1994, 0.2010])])

# 测试时，无需对图像做标准化以外的增强数据处理。
transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                      [0.2023, 0.1994, 0.2010])])

def train_test_data(data_dir, input_dir, batch_size=128):
    # 读取原始图像文件。flag=1 说明输入图像有三个通道（彩色）。
    train_ds = gdata.vision.ImageFolderDataset(
        os.path.join(data_dir, input_dir, 'train'), flag=1)
    test_ds = gdata.vision.ImageFolderDataset(
        os.path.join(data_dir, input_dir, 'test'), flag=1)

    train_data = gdata.DataLoader(train_ds.transform_first(transform_train),
                                batch_size, shuffle=True, last_batch='keep')
    test_data = gdata.DataLoader(test_ds.transform_first(transform_test),
                                batch_size, shuffle=False, last_batch='keep')
    return train_data, test_data

loss = gloss.SoftmaxCrossEntropyLoss()

def get_loss(data, net, ctx):
    l = 0.0
    for X, y in data:
        y = y.as_in_context(ctx)
        # 计算预训练模型输出层的输入，即特征
        output_features = net.features(X.as_in_context(ctx))
        # 将特征作为我们定义的输出网络的输入，计算输出
        # outputs = net.output_new(output_features)
        outputs = output_features
        l += loss(outputs, y).mean().asscalar()
    return l / len(data)


def train(net , train_data, test_data, num_epochs, lr, wd, ctx, lr_period, lr_decay, batch_size=128):
    # 只训练我们定义的输出网络
    # trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd', {'learning_rate':lr, 'momentum':0.9, 'wd':wd})
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr, 'momentum':0.9, 'wd':wd})

    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_l = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for X, y in train_data:
            y = y.astype('float32').as_in_context(ctx)
            #计算预训练模型输出层的输入，即特征
            output_features = net.features(X.as_in_context(ctx))
            with autograd.record():
                #将特征作为我们定义的输出网络的输入，计算输出
                # outputs = net.output_new(output_features)
                # l = loss(outputs, y)
                l = loss(output_features, y)
            # 反向传播只发生在我们定义的输出网络上
            l.backward()
            trainer.step(batch_size)
            train_l += l.mean().asscalar()
        cur_time = datetime.datetime().now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_s = "time %02d:%02d%02d" % (h, m, s)
        if test_data is not None:
            test_loss = get_loss(test_data, net, ctx)
            epoch_s = ("epoch %d, train_loss %f, test loss %f, "
                    % (epoch + 1, train_l / len(train_data), test_loss))
        else:
            epoch_s = ("epoch %d, train loss %f, "
                    % (epoch + 1, train_l / len(train_data)))
        prev_time = cur_time
        print(epoch_s + time_s + ', lr ' + str(trainer.learning_rate))


def get_net(ctx):
    net = nn.Sequential(get_shufflenet())
    net.initialize(init.Xavier(), ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    return net


def unzip_dataset(path, zip_file):
    with zipfile.ZipFile(os.path.join(path, zip_file), 'r') as z:
        z.extractall(path)

if __name__ == "__main__":
    data_dir = "E:/srcs/anti-spoofing/data"
    input_dir = 'nir_flow_1w'
    # zip_file = input_dir + '.zip'
    #unzip_dataset(data_dir, zip_file)

    train_data, test_data = train_test_data(data_dir, input_dir)
    ctx, num_epochs, lr, wd = gb.try_gpu(), 1, 0.01, 1e-4
    lr_period, lr_decay, net = 10, 0.1, get_net(ctx)
    train(net, train_data, test_data, num_epochs, lr, wd, ctx, lr_period, lr_decay)