# -*- coding:utf-8- *-

import os,sys

# 修改文件内容并保存
def transform_relImgPath2NewPath(main_dir, rel_path, new_path):
    nir_img_anno_filename = os.path.join(main_dir+'/anno.txt')
    nir_img_prob_filename = os.path.join(main_dir+'/prob.txt')
    nir_img_train_filename = os.path.join(main_dir+'/train.txt')

    with open(nir_img_anno_filename, 'r+') as f:
        lines = f.readlines()
        f.seek(0,0) # 将文件读写指针移位到文件开始位置
        for line in lines:
            line_new = line.replace(rel_path, new_path)
            f.write(line_new)

    with open(nir_img_prob_filename, 'r+') as f:
        lines = f.readlines()
        f.seek(0,0) # 将文件读写指针移位到文件开始位置
        for line in lines:
            line_new = line.replace(rel_path, new_path)
            f.write(line_new)

    with open(nir_img_train_filename, 'r+') as f:
        lines = f.readlines()
        f.seek(0,0) # 将文件读写指针移位到文件开始位置
        for line in lines:
            line_new = line.replace(rel_path, new_path)
            f.write(line_new)

#修改文件内容，并写入新文件
def transform2_relImgPath2NewPath(main_dir, main_dir_new, trainlist_dir_new, rel_path, new_path):
    nir_img_anno_filename = os.path.join(main_dir+'/anno.txt')
    nir_img_anno_filename_new = os.path.join(main_dir_new+'/anno.txt')
    nir_img_prob_filename = os.path.join(main_dir+'/prob.txt')
    nir_img_prob_filename_new = os.path.join(main_dir_new+'/prob.txt')
    nir_img_train_filename = os.path.join(main_dir+'/train.txt')
    nir_img_train_filename_new = os.path.join(trainlist_dir_new+'/train.txt')

    results = []
    with open(nir_img_anno_filename, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            line_new = line.replace(rel_path, new_path)
            results.append(line_new)
    
    with open(nir_img_anno_filename_new, 'w+') as f:
        for line in results:
            f.write(line)

    results1 = []
    with open(nir_img_prob_filename, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            line_new = line.replace(rel_path, new_path)
            results1.append(line_new)
    
    with open(nir_img_prob_filename_new, 'w+') as f:
        for line in results1:
            f.write(line)

    results2 = []
    with open(nir_img_train_filename, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            line_new = line.replace(rel_path, new_path)
            results2.append(line_new)
    
    with open(nir_img_train_filename_new, 'w+') as f:
        for line in results2:
            f.write(line)


if __name__ == '__main__':
    main_dir = '../data/.../images/nirs/'
    main_dir_new = './.../'
    trainlist_dir_new = '../data/mtcnn/imglists'
    rel_path = './xxx'
    new_path = '/xxx'
    # transform_relImgPath2NewPath(main_dir, rel_path, new_path)
    transform2_relImgPath2NewPath(main_dir, main_dir_new, trainlist_dir_new, rel_path, new_path)
    
    