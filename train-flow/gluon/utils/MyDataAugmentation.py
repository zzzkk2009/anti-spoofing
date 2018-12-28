# -*-coding:utf-8-*-

from mxnet import image
from mxnet import nd

def transform_train(data, label):
    im = image.imresize(data.astype('float32') / 255, 256, 256)
    auglist = image.CreateAugmenter(data_shape=(3, 256, 256), resize=0,
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=None, std=None,
                        brightness=0, contrast=0,
                        saturation=0, hue=0,
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2, 0, 1))
    return (im, nd.array([label]).asscalar().astype('float32'))


def transform_test(data, label):
    im = image.imresize(data.astype('float32') / 255, 256, 256)
    im = nd.transpose(im, (2, 0, 1))  # 之前没有运行此变换
    return (im, nd.array([label]).asscalar().astype('float32'))
