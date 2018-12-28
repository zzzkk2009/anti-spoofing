# -*- coding:utf-8 -*-

from mxnet import autograd, nd, gluon, init
from mxnet.gluon import loss as gloss
import random
from models.shufflenet_v2 import ShuffleNetV2

# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j) # take函数根据索引返回对应元素

# 定义损失函数
def get_loss():
    loss = gloss.SoftmaxCrossEntropyLoss()
    return loss

# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

# 定义优化算法(gluon)
def get_trainer(lr, momentum=0.9, wd=0.001):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    return trainer

# 计算分类准确率
# y_hat:预测概率分布；y：标签
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

#评估模型准确率
def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)

# 训练模型
def train(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None,
         wd, ctx, lr_period, lr_decay, trainer):
    for epoch in range(num_epochs):
        train_l_sum = 0
        train_acc_sum = 0
        start = time.time()
        prev_time = datetime.datetime.now()
        if epoch > 0 and epoch % lr_period == 0:
            if trainer is None:
                trainer.set_learning_rate(trainer.learning_rate * lr_decay)

        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        print(time_str)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, lr %s' % (
            epoch + 1, train_l_sum / len(train_iter), train_acc_sum / len(train_iter), test_acc, str(trainer.learning_rate)
        ))
        # net.collect_params().save('./model/alexnet.params')

if __name__ == '__main__':

    net = ShuffleNetV2(num_classes=2, width_multiplier=0.5)
    net.initialize(mx.init.Xavier())