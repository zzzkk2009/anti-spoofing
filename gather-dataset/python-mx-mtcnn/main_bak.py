# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

def testimg(detector):

    img = cv2.imread('../imgs/oscar.jpg')

    t1 = time.time()
    results = detector.detect_face(img)
    print 'time: ',time.time() - t1

    if results is not None:

        total_boxes = results[0]
        points = results[1]

        draw = img.copy()
        for b in total_boxes:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

        for p in points:
            for i in range(5):
                cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

        cv2.imshow("detection result", draw)
        cv2.imwrite("../result.png", draw)
        cv2.waitKey(0)

# --------------
# test on camera
# --------------
def testcamera(detector):

    camera = cv2.VideoCapture(0)
    while True:
        grab, frame = camera.read()
        img = cv2.resize(frame, (640,480))
       
        t1 = time.time()
        results = detector.detect_face(img)
        print 'time: ',time.time() - t1

        if results is None:
            cv2.imshow("detection result", img)
            cv2.waitKey(1)
            continue

        total_boxes = results[0]
        points = results[1]

        draw = img.copy()
        for b in total_boxes:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

        for p in points:
            for i in range(5):
                cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
        cv2.imshow("detection result", draw)
        key=cv2.waitKey(1)
        if 'q'==chr(key & 255) or 'Q'==chr(key & 255):
            break;
       

if __name__=="__main__":
    detector = MtcnnDetector(model_folder='../model/mxnet', ctx=mx.gpu(0), num_worker = 4 , accurate_landmark = False)
    #testimg(detector)
    testcamera(detector)