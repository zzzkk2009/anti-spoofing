# -*- coding: utf-8 -*-

import os
import shutil
import zipfile
import zlib


def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

def reorg_train_test_flow_data2(data_dir, processed_dir, labels, start_dirs, end_dirs, org_dist_lDir_diffs):

    # print("data_dir:", data_dir)
    mkdir_if_not_exist([processed_dir])

    for _, label in enumerate(labels):

        start_dir = start_dirs[int(label)]
        end_dir = end_dirs[int(label)]
        org_dist_lDir_diff = org_dist_lDir_diffs[int(label)]

        for _, l_dir in enumerate(os.listdir(os.path.join(data_dir, label))):

            i_l_dir = int(l_dir)

            if -1 != start_dir and i_l_dir < start_dir:
                continue

            if -1 != end_dir and i_l_dir > end_dir:
                break

            dist_dir = str(i_l_dir + int(org_dist_lDir_diff))
            dist_dir = dist_dir.zfill(5)

            mkdir_if_not_exist([processed_dir, str(label), dist_dir])
           
            for j, l_file in enumerate(os.listdir(os.path.join(data_dir, str(label), l_dir))):
                if not os.path.exists(os.path.join(processed_dir, str(label), dist_dir, l_file)):
                            shutil.copy(os.path.join(data_dir, str(label), l_dir, l_file), 
                                os.path.join(processed_dir, str(label), dist_dir))

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
    ori_dir = main_dir + "/nir-fld-mtcnn/nir-fld-mtcnn/data/48x64/limitedArea/nirs"
    new_dir = main_dir + "/data/nir_flow_48x64_limitedarea_1.3w"
    labels = ['0', '1'] # ['0', '1']
    start_dirs = [1, 1]
    end_dirs = [10, 3]
    org_dist_lDir_diffs = [0, 0]
    reorg_train_test_flow_data2(ori_dir, new_dir, labels, start_dirs, end_dirs, org_dist_lDir_diffs)
    #make_zip(new_dir, new_dir + '.zip')