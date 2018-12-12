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

main_path = './'
videos_save_path = os.path.join(main_path, 'videos')
videos_rgbs_save_path = os.path.join(videos_save_path, 'rgbs')
videos_rgbs_1_save_path = os.path.join(videos_save_path, 'rgbs', '1')
videos_rgbs_0_save_path = os.path.join(videos_save_path, 'rgbs', '0')
videos_nirs_save_path = os.path.join(videos_save_path, 'nirs')
videos_nirs_1_save_path = os.path.join(videos_save_path, 'nirs', '1')
videos_nirs_0_save_path = os.path.join(videos_save_path, 'nirs', '0')

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
    samplingPositive = True
    gatheredFrames = 0 # 采集的帧数
    videoWriter_rgb = None
    videoWriter_ir = None
    frameCount = 0

    writePath_rgb = videos_rgbs_1_save_path if samplingPositive else videos_rgbs_0_save_path
    writePath_nir = videos_nirs_1_save_path if samplingPositive else videos_nirs_0_save_path

    while True:
        start = time.time()
        frameCount += 1
        grab_rgb, frame_rgb = camera_rgb.read() # 第一个返回值grab代表是否读取到图片
        grab_ir, frame_ir = camera_ir.read()

        end = time.time()
        seconds = end - start if end - start > 0 else -1
        fps  = int(1 / seconds)

        if starting:
            if videoWriter_rgb is None or gatheredFrames > 30 * 30: # 每个视频30秒
                print('gatheredFrames==', gatheredFrames)
                gatheredFrames = 0

                videoWriter_rgb = cv2.VideoWriter(writePath_rgb  + '/' + str(int(start)) + '.mp4', fourcc, 30.0, size)
                videoWriter_ir = cv2.VideoWriter(writePath_nir  + '/' + str(int(start)) + '.mp4', fourcc, 30.0, size)
            
            videoWriter_rgb.write(frame_rgb) #写视频帧    
            videoWriter_ir.write(frame_ir) #写视频帧
            gatheredFrames += 1
        
        cv2.putText(frame_rgb,'FPS:'+str(fps),(25,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        cv2.putText(frame_rgb,'starting:'+str(starting)+',samplingPositive:'+str(samplingPositive),(25,size[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        cv2.imshow("detection result rgb", frame_rgb)
        cv2.imshow("detection result ir", frame_ir)

        key=cv2.waitKey(1)
        if 'q'==chr(key & 255) or 'Q'==chr(key & 255) or 27 == key:
            break
        
        if 'p' == chr(key & 255):
            samplingPositive = not samplingPositive
            writePath_rgb = videos_rgbs_1_save_path if samplingPositive else videos_rgbs_0_save_path
            writePath_nir = videos_nirs_1_save_path if samplingPositive else videos_nirs_0_save_path
            videoWriter_rgb = None
        
        if 32 == key: #空格
            starting = not starting

    # 释放VideoCapture对象
    camera_rgb.release()
    camera_ir.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    utils.makedir_if_not_exist(videos_save_path)
    utils.makedir_if_not_exist(videos_rgbs_save_path)
    utils.makedir_if_not_exist(videos_nirs_save_path)
    utils.makedir_if_not_exist(videos_rgbs_1_save_path)
    utils.makedir_if_not_exist(videos_rgbs_0_save_path)
    utils.makedir_if_not_exist(videos_nirs_1_save_path)
    utils.makedir_if_not_exist(videos_nirs_0_save_path)
    testcamera_nir()
