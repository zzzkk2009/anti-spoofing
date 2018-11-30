# -*- coding:utf-8 -*-

import os
import shutil
import zipfile
import zlib


def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))


def reorg_train_test_flow_data(data_dir, processed_dir, labels, test_ratio=0.2):

    # print("data_dir:", data_dir)
    mkdir_if_not_exist([processed_dir])

    sampleMargin = int(1 / test_ratio)
    for _, label in enumerate(labels):

        # mkdir_if_not_exist([processed_dir, str(label)])
        train_dir = 'train'
        test_dir = 'test'
        mkdir_if_not_exist([processed_dir, train_dir, str(label)])
        mkdir_if_not_exist([processed_dir, test_dir, str(label)])

        for _, l_dir in enumerate(os.listdir(os.path.join(data_dir, label))):
            mkdir_if_not_exist([processed_dir, train_dir, str(label), l_dir])
            mkdir_if_not_exist([processed_dir, test_dir, str(label), l_dir])
            for j, l_file in enumerate(os.listdir(os.path.join(data_dir, label, l_dir))):
                if j % sampleMargin == 0:
                    if not os.path.exists(os.path.join(processed_dir, test_dir, label, l_dir, l_file)):
                        shutil.copy(os.path.join(data_dir, label, l_dir, l_file), 
                            os.path.join(processed_dir, test_dir, str(label), l_dir))
                else:
                    if not os.path.exists(os.path.join(processed_dir, train_dir, label, l_dir, l_file)):
                        shutil.copy(os.path.join(data_dir, label, l_dir, l_file), 
                            os.path.join(processed_dir, train_dir, str(label), l_dir))


def make_zip(src_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, mode='w')
    pre_len = len(os.path.dirname(src_dir))
    for parent, dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep) #相对路径
            zipf.write(pathfile, arcname, compress_type=zipfile.ZIP_DEFLATED)
    zipf.close()

if __name__ == "__main__":
    main_dir = "E:/srcs/anti-spoofing"
    ori_dir = main_dir + "/nir-fld-mtcnn/nir-fld-mtcnn/data/c48x64"
    new_dir = main_dir + "/data/nir_flow_c48x64_1w5k"
    labels = ['0', '1'] # ['0', '1']
    reorg_train_test_flow_data(ori_dir, new_dir, labels)
    #make_zip(new_dir, new_dir + '.zip')
