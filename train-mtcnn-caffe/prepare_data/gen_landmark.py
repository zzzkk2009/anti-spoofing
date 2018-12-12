import numpy as np
import cv2
import threading
import argparse
import math
import os,sys
import numpy.random as npr
from utils import IoU
sys.path.append(os.getcwd())
from config import config
import tools.image_processing as image_processing

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.landmark_names = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.landmark_names
        except Exception:
            return None

def gen_landmark_minibatch_thread(size, start_idx, annotation_lines, landmark_lines, imdir, landmark_save_dir, base_num):
    num_images = len(annotation_lines)
    landmark_names = list()
    for i in range(num_images):
        cur_annotation_line = annotation_lines[i].strip().split()
        cur_landmark_line = landmark_lines[i].strip().split()
        im_path = cur_annotation_line[0]
        bbox = map(float, cur_annotation_line[1:])
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        landmark = map(float, cur_landmark_line[1:])
        landmarks = np.array(landmark, dtype=np.float32).reshape(-1, 10)
        img = cv2.imread(os.path.join(imdir, im_path))
        cur_landmark_names = gen_landmark_for_one_image(size, start_idx+i, img, landmark_save_dir, boxes, landmarks, base_num)
        landmark_names = landmark_names + cur_landmark_names


    return landmark_names


def gen_landmark_minibatch(size, start_idx, annotation_lines, landmark_lines, imdir, landmark_save_dir, base_num, thread_num = 4):
    num_images = len(annotation_lines)
    num_per_thread = math.ceil(float(num_images)/thread_num)
    threads = []
    for t in range(thread_num):
        cur_start_idx = int(num_per_thread*t)
        cur_end_idx = int(min(num_per_thread*(t+1),num_images))
        cur_annotation_lines = annotation_lines[cur_start_idx:cur_end_idx]
        cur_landmark_lines = landmark_lines[cur_start_idx:cur_end_idx]
        cur_thread = MyThread(gen_landmark_minibatch_thread,(size, start_idx+cur_start_idx, cur_annotation_lines, cur_landmark_lines, 
                                                        imdir, landmark_save_dir, base_num))
        threads.append(cur_thread)
    for t in range(thread_num):
        threads[t].start()

    landmark_names = list()
    
    for t in range(thread_num):
        cur_landmark_names = threads[t].get_result()
        landmark_names = landmark_names + cur_landmark_names

    return landmark_names
	

def gen_landmark_for_one_image(size, idx, img, landmark_save_dir,boxes, landmarks, base_num = 1):
    landmark_names = list()
    landmark_num = 0
    
    width = img.shape[1]
    height = img.shape[0]
    
    box_num = boxes.shape[0]
    for bb in range(box_num):
        box = boxes[bb]
        landmark = landmarks[bb]
        x1, y1, w, h = box
 
        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        #angles = [0,-15,-30,-45,15,30,45]
        angles = [0]
        rot_num = len(angles)
        for rr in range(rot_num):
            #print(landmark)
            rot_img, rot_landmark = image_processing.rotateWithLandmark(img, landmark, angles[rr],1)
            rot_x1, rot_y1 = x1, y1 # as the angle is not large, bbox is almost the same
            for i in range(base_num):
                cur_size = npr.randint(int(min(w, h) * 0.9), np.ceil(1.11 * max(w, h)))

                # delta here is the offset of box center
                delta_x = npr.randint(-w * 0.15, w * 0.15)
                delta_y = npr.randint(-h * 0.15, h * 0.15)

                nx1 = int(max(x1 + w / 2 + delta_x - cur_size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - cur_size / 2, 0))
                nx2 = nx1 + cur_size
                ny2 = ny1 + cur_size

                if nx2 > width or ny2 > height:
                    continue
                ignore = 0
                max_x_landmark = -1
                min_x_landmark = width+1
                max_y_landmark = -1
                min_y_landmark = height+1
                for j in range(5):
                    if rot_landmark[j*2] < nx1 or rot_landmark[j*2] >= nx1 + cur_size:
                        ignore = 1
                    if rot_landmark[j*2+1] < ny1 or rot_landmark[j*2+1] > ny1 + cur_size:
                        ignore = 1
                    if max_x_landmark < rot_landmark[j*2]:
                        max_x_landmark = rot_landmark[j*2]
                    if min_x_landmark > rot_landmark[j*2]:
                        min_x_landmark = rot_landmark[j*2]
                    if max_y_landmark < rot_landmark[j*2+1]:
                        max_y_landmark = rot_landmark[j*2+1]
                    if min_y_landmark > rot_landmark[j*2+1]:
                        min_y_landmark = rot_landmark[j*2+1]
												
                if ignore == 1:
                    continue
                if (max_x_landmark - min_x_landmark < 0.2*cur_size) and (max_y_landmark-min_y_landmark < 0.2*cur_size):
                    continue
                offset_x1 = (rot_landmark[0] - nx1 + 0.5) / float(cur_size)
                offset_y1 = (rot_landmark[1] - ny1 + 0.5) / float(cur_size)
                offset_x2 = (rot_landmark[2] - nx1 + 0.5) / float(cur_size)
                offset_y2 = (rot_landmark[3] - ny1 + 0.5) / float(cur_size)
                offset_x3 = (rot_landmark[4] - nx1 + 0.5) / float(cur_size)
                offset_y3 = (rot_landmark[5] - ny1 + 0.5) / float(cur_size)
                offset_x4 = (rot_landmark[6] - nx1 + 0.5) / float(cur_size)
                offset_y4 = (rot_landmark[7] - ny1 + 0.5) / float(cur_size)
                offset_x5 = (rot_landmark[8] - nx1 + 0.5) / float(cur_size)
                offset_y5 = (rot_landmark[9] - ny1 + 0.5) / float(cur_size)
               
                cropped_im = rot_img[ny1 : ny2, nx1 : nx2, :]
                resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)
                save_file = '%s/%d_%d.jpg'%(landmark_save_dir,idx,landmark_num)
                if cv2.imwrite(save_file, resized_im):
                    line = '%s/%d_%d'%(landmark_save_dir,idx,landmark_num) + ' -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f'%(
                                             offset_x1, offset_x2, offset_x3, offset_x4, offset_x5, 
                                             offset_y1, offset_y2, offset_y3, offset_y4, offset_y5)
                    landmark_names.append(line)
                    landmark_num += 1
				
    return landmark_names

def gen_landmark(size=20, base_num = 1, thread_num = 4):
    anno_file = "%s/prepare_data/celeba_annotations/list_bbox_celeba.txt"%config.root
    landmark_file = "%s/prepare_data/celeba_annotations/list_landmarks_celeba.txt"%config.root
    imdir = "%s/data/img_celeba"%config.root
    landmark_save_dir = "%s/prepare_data/%d/landmark"%(config.root,size)
    
    save_dir = "%s/prepare_data/%d"%(config.root,size)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(landmark_save_dir):
        os.mkdir(landmark_save_dir)
    f1 = open(os.path.join(save_dir, 'landmark.txt'), 'w')

    with open(anno_file, 'r') as f:
        annotation_lines = f.readlines()
    with open(landmark_file, 'r') as f:
        landmark_lines = f.readlines()

    num = len(annotation_lines)-2
    print "%d pics in total" % num
    annotation_lines = annotation_lines[2:]
    landmark_lines = landmark_lines[2:]
    batch_size = thread_num*10
    landmark_num = 0
    start_idx = 0
    while start_idx < num:
        end_idx = min(start_idx+batch_size,num)
        cur_annotation_lines = annotation_lines[start_idx:end_idx]
        cur_landmark_lines = landmark_lines[start_idx:end_idx]
        landmark_names = gen_landmark_minibatch(size, start_idx, cur_annotation_lines, cur_landmark_lines, 
                                            imdir, landmark_save_dir, base_num, thread_num)
        cur_landmark_num = len(landmark_names)
        for i in range(cur_landmark_num):
            f1.write(landmark_names[i]+'\n')
        landmark_num += cur_landmark_num
        start_idx = end_idx
        print '%s images done, landmark: %d'%(end_idx,landmark_num)

    f1.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train proposal net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size', dest='size', help='20 or 24 or 48', default='20', type=str)
    parser.add_argument('--base_num', dest='base_num', help='base num', default='1', type=str)
    parser.add_argument('--thread_num', dest='thread_num', help='thread num', default='4', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    gen_landmark(int(args.size), int(args.base_num), int(args.thread_num))