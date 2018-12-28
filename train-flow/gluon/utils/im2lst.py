#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 使用示例：
# 注：im2rec.py文件必须和图片目录放在同级目录
# prefix : train / test (基于im2lst.py文件目录的train目录或test目录)
# root : train / test
# python im2lst.py --recursive --exts=.png

# .lst文件样例：第一列是图像路径，第二列是label
# python im2lst.py --recursive --exts=.png --train-ratio=0.8

from __future__ import print_function
import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time
import traceback

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

def list_image(root, recursive, exts):
    """Traverses the root of directory that contains images and
    generates image list iterator.
    Parameters
    ----------
    root: string
    recursive: bool
    exts: string
    Returns
    -------
    image iterator that contains all the image under the specified path
    """

    i = 0
    print('recursive==', recursive)
    if recursive:
        cat = {}
        labels = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    relpath = os.path.relpath(fpath, root)
                    label = int(relpath.split('\\', 1)[0])
                    if path not in cat:
                        cat[path] = label
                    yield (i, relpath, cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            relpath = os.path.relpath(k, root)
            # print('relpath', relpath)
            label = relpath.split('\\', 1)[0]
            # print('label', label)
            print(relpath, v)
            # print(os.path.relpath(k, root), v)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1

def write_list(path_out, image_list):
    """Hepler function to write image list into the file.
    The format is as below,
    integer_image_index \t float_label_index \t path_to_image
    Note that the blank between number and tab is only used for readability.
    Parameters
    ----------
    path_out: string
    image_list: list
    """
    filename = os.path.join(args.root, path_out)
    print('filename=', filename)
    with open(filename, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%s\t' % item[1]
            line += '%f\n' % item[2]
            fout.write(line)

def make_list(args):
    """Generates .lst file.
    Parameters
    ----------
    args: object that contains all the arguments
    """
    image_list = list_image(args.root, args.recursive, args.exts)
    image_list = list(image_list)
    if args.shuffle is True:
        random.seed(100)
        random.shuffle(image_list)
    N = len(image_list)
    chunk_size = (N + args.chunks - 1) // args.chunks

    _prefix = args.prefix
    if not args.prefix:
        # os.path.split('C:/soft/python/test') -> ('C:/soft/python', 'test')
        _prefix = os.path.split(args.root)[1] 

    print('_prefix=', _prefix)

    for i in range(args.chunks):
        chunk = image_list[i * chunk_size:(i + 1) * chunk_size]
        if args.chunks > 1:
            str_chunk = '_%d' % i
        else:
            str_chunk = ''
        sep = int(chunk_size * args.train_ratio)
        sep_test = int(chunk_size * args.test_ratio)
        if args.train_ratio == 1.0:
            write_list(_prefix + str_chunk + '.lst', chunk)
        else:
            if args.test_ratio:
                write_list(_prefix + str_chunk + '_test.lst', chunk[:sep_test])
            if args.train_ratio + args.test_ratio < 1.0:
                write_list(_prefix + str_chunk + '_val.lst', chunk[sep_test + sep:])
            write_list(_prefix + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep])

def parse_args():
    """Defines all arguments.
    Returns
    -------
    args object that contains all the params
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--root', type=str, help='path to folder containing images.',
        default='E:/srcs/anti-spoofing/dataset/nir_flow_48x64_limitedarea_1.3w')
    cgroup.add_argument('--prefix', help='prefix of input/output lst files.')
    cgroup.add_argument('--exts', nargs='+', default=['.png'], # '.jpeg', '.jpg', 
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    cgroup.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of images to use for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0,
                        help='Ratio of images to use for testing.')
    cgroup.add_argument('--recursive', action='store_true', default=True,
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')
    cgroup.add_argument('--recursive-label-n', type=int, default=1, help='root folder下面第一层文件夹作为label标签')
    cgroup.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                        help='If this is passed, \
        im2rec will not randomize the image order in <prefix>.lst')
    args = parser.parse_args()
    #args.prefix = os.path.abspath(args.prefix)
    #args.root = os.path.abspath(args.root)
    return args

if __name__ == '__main__':
    args = parse_args()
    print('make_list start...')
    make_list(args)
    print('make_list ok!')
        
    