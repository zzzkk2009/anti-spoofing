# -*- coding:utf-8- *-
import sys
# sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import numpy as np
import os
import time
import utils

# ./
#   images/
#          rgbs/
#                orgs/
#                     rgb_org_xxxx.png
#                     ...
#                rects/
#                     rgb_rect_xxxx.png
#                     ...
#                anns/
#                     rgb_org_xxxx.xml
#                     rgb_org_xxxx.xml.txt
#                     ...
#          nirs/
#                orgs/
#                     nir_org_xxxx.png
#                     ...
#                rects/
#                     nir_rect_xxxx.png
#                     ...
#                anns/
#                     nir_org_xxxx.xml
#                     nir_org_xxxx.xml.txt
#                     ...
#   videos/
#         
#

main_path = './'
datetime = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
sub_path = os.path.join(main_path, datetime)

# images_rgbs_orgs_save_path = main_path + datetime + '/images/rgbs/orgs'
images_save_path = os.path.join(sub_path, 'images')
images_rgbs_save_path = os.path.join(sub_path, 'images', 'rgbs')
images_rgbs_orgs_save_path = os.path.join(sub_path, 'images', 'rgbs', 'orgs')
images_rgbs_rects_save_path = os.path.join(sub_path, 'images', 'rgbs', 'rects')
images_rgbs_anns_save_path = os.path.join(sub_path, 'images', 'rgbs', 'anns')

# utils.mkdir_recursively(images_rgbs_orgs_save_path)

images_nirs_save_path = os.path.join(sub_path, 'images', 'nirs')
images_nirs_orgs_save_path = os.path.join(sub_path, 'images', 'nirs', 'orgs')
images_nirs_rects_save_path = os.path.join(sub_path, 'images', 'nirs', 'rects')
images_nirs_anns_save_path = os.path.join(sub_path, 'images', 'nirs', 'anns')

# videos_save_path = os.path.join(sub_path, 'videos')


thresh = [0.9, 0.6, 0.7]
min_face_size = 24
stride = 2
slide_window = False
detectors = [None, None, None]
prefix = ['./data/MTCNN_model/PNet_landmark/PNet', './data/MTCNN_model/RNet_landmark/RNet', './data/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1])
detectors[1] = RNet
ONet = Detector(O_Net, 48, 1, model_path[2])
detectors[2] = ONet
mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

# --------------
# test on camera
# --------------
def testcamera_nir():

    camera_rgb = cv2.VideoCapture(0)
    camera_ir = cv2.VideoCapture(1)

    frameCount = 0
    prevSaveDetectedFaceFrameCount = 0 #上一次保存检测到人脸是第几帧
    while True:
        start = time.time()
        frameCount += 1
        grab_rgb, frame_rgb = camera_rgb.read() # 第一个返回值grab代表是否读取到图片
        grab_ir, frame_ir = camera_ir.read()

        image_rgb = np.array(frame_rgb)
        boxes_c_rgb, landmarks_rgb = mtcnn_detector.detect(image_rgb)

        end = time.time()
        seconds = end - start
        fps  = int(1 / seconds)

        if 0 == boxes_c_rgb.size:
            cv2.putText(frame_rgb,'FPS:'+str(fps),(25,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            cv2.imshow("detection result rgb", frame_rgb)
            cv2.imshow("detection result ir", frame_ir)
            cv2.waitKey(1)
            continue
        
        total_boxes_rgb = []
        for i in range(boxes_c_rgb.shape[0]):
            bbox = boxes_c_rgb[i, :4]
            score = boxes_c_rgb[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            total_boxes_rgb.append(corpbbox)
        
        if (frameCount - prevSaveDetectedFaceFrameCount) >= int(fps * 2): # 每2秒保存一帧
            rgb_img_filename = os.path.join(images_rgbs_orgs_save_path, 'rgb_org_%04d.png' % frameCount)
            ir_img_filename = os.path.join(images_nirs_orgs_save_path, 'nir_org_%04d.png' % frameCount)
            cv2.imwrite(rgb_img_filename, frame_rgb)
            cv2.imwrite(ir_img_filename, frame_ir)

            draw_rgb = frame_rgb.copy()
            draw_ir = frame_ir.copy()

            rgb_img_anns_filename = os.path.join(images_rgbs_anns_save_path, 'rgb_org_%04d.xml' % frameCount)
            ir_img_anns_filename = os.path.join(images_nirs_anns_save_path, 'nir_org_%04d.xml' % frameCount)
            # print('rgb_img_anns_filename===', rgb_img_anns_filename)
            utils.saveXML(rgb_img_anns_filename, total_boxes_rgb, 0, frame_rgb.shape[1], frame_rgb.shape[0])
            utils.saveXML(ir_img_anns_filename, total_boxes_rgb, 0, frame_ir.shape[1], frame_ir.shape[0], True)

            for b in total_boxes_rgb:
                p1 = (int(b[0]), int(b[1]))
                p2 = (int(b[2]), int(b[3]))
                cv2.rectangle(draw_rgb, p1, p2, (0, 255, 0))

                # p1_ir = (int(b[0])+20, int(b[1])-10)
                # p2_ir = (int(b[2])+20, int(b[3])-10)
                p1_ir = p1
                p2_ir = p2
                cv2.rectangle(draw_ir, p1_ir, p2_ir, (0, 255, 0))

            rgb_img_rect_filename = os.path.join(images_rgbs_rects_save_path, 'rgb_rect_%04d.png' % frameCount)
            ir_img_rect_filename = os.path.join(images_nirs_rects_save_path, 'nir_rect_%04d.png' % frameCount)
            cv2.imwrite(rgb_img_rect_filename, draw_rgb)
            cv2.imwrite(ir_img_rect_filename, draw_ir)

            prevSaveDetectedFaceFrameCount = frameCount
            
            cv2.putText(draw_rgb,'FPS:'+str(fps),(25,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            cv2.imshow("detection result rgb", draw_rgb)
            cv2.imshow("detection result ir", draw_ir)
        key=cv2.waitKey(1)
        if 'q'==chr(key & 255) or 'Q'==chr(key & 255) or 27 == key: # 27:Esc
            break

    # 释放VideoCapture对象
    camera_rgb.release()
    camera_ir.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    utils.makedir_if_not_exist(sub_path)
    # print('datetime===', datetime)
    
    utils.makedir_if_not_exist(images_save_path)
    utils.makedir_if_not_exist(images_rgbs_save_path)
    utils.makedir_if_not_exist(images_rgbs_orgs_save_path)
    utils.makedir_if_not_exist(images_rgbs_rects_save_path)
    utils.makedir_if_not_exist(images_rgbs_anns_save_path)

    utils.makedir_if_not_exist(images_nirs_save_path)
    utils.makedir_if_not_exist(images_nirs_orgs_save_path)
    utils.makedir_if_not_exist(images_nirs_rects_save_path)
    utils.makedir_if_not_exist(images_nirs_anns_save_path)

    testcamera_nir()