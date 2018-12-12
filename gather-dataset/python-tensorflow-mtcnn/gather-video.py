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
videos_save_path = os.path.join(main_path, 'videos')


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

    #获得码率及尺寸
    _fps = camera_rgb.get(cv2.CAP_PROP_FPS)
    size = (int(camera_rgb.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(camera_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('_fps===', _fps)
    print('size===', size)

    #指定写视频的格式, I420-avi, MJPG-mp4
    #fourcc是一种编码格式，我们保存视频时要指定文件名、编码格式、FPS、输出尺寸、颜色模式
    # cv2.VideoWriter_fourcc('I','4','2','0')，YUV颜色编码，AVI格式输出。
    # cv2.VideoWriter_fourcc('P','I','M','1')，MPEG-1编码，AVI格式输出。
    # cv2.VideoWriter_fourcc('X','V','I','D')，MPEG-4编码，AVI格式输出。
    # cv2.VideoWriter_fourcc('F','L','V','1')，Flash编码，flv格式输出。
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    

    starting = False
    gatheredFrames = 0 # 采集的帧数
    videoWriter = None
    frameCount = 0
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

        if boxes_c_rgb.size > 0:
            if (not starting) or (starting and gatheredFrames > 30 * 10):
                print('gatheredFrames==', gatheredFrames)
                starting = not starting
                gatheredFrames = 0
                videoWriter = cv2.VideoWriter(videos_save_path + '/' + str(int(start)) + '.mp4', fourcc, 30.0, size)
                
            videoWriter.write(frame_ir) #写视频帧
            gatheredFrames += 1
        else:
            starting = False
            gatheredFrames = 0

        total_boxes_rgb = []
        for i in range(boxes_c_rgb.shape[0]):
            bbox = boxes_c_rgb[i, :4]
            score = boxes_c_rgb[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            total_boxes_rgb.append(corpbbox)
        
        draw_rgb = frame_rgb.copy()
        draw_ir = frame_ir.copy()
        for b in total_boxes_rgb:
            p1 = (int(b[0]), int(b[1]))
            p2 = (int(b[2]), int(b[3]))
            cv2.rectangle(draw_rgb, p1, p2, (0, 255, 0))
            p1_ir = (int(b[0])+20, int(b[1])-10)
            p2_ir = (int(b[2])+20, int(b[3])-10)
            cv2.rectangle(draw_ir, p1_ir, p2_ir, (0, 255, 0))
        
        cv2.putText(draw_rgb,'FPS:'+str(fps),(25,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        cv2.putText(draw_rgb,'starting:'+str(starting),(25,size[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        cv2.imshow("detection result rgb", draw_rgb)
        cv2.imshow("detection result ir", draw_ir)

        key=cv2.waitKey(1)
        if 'q'==chr(key & 255) or 'Q'==chr(key & 255) or 27 == key:
            break

    # 释放VideoCapture对象
    camera_rgb.release()
    camera_ir.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    utils.makedir_if_not_exist(videos_save_path)
    testcamera_nir()