#ifndef __ZK_HEAD_POSE_ESTIMATION_H__
#define __ZK_HEAD_POSE_ESTIMATION_H__

#include <iostream>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <vector>
#include <map>
#include <mtcnn/mtcnn.h>

using namespace cv;
using namespace std;

namespace hpe {
	typedef struct HeadPose {
		float yaw; // Yaw:摇头  左正右负
		float pitch; // 点头 上负下正
		float roll; //摆头（歪头）左负 右正
	} HeadPose;
	void rot2Euler(const Mat& rotation3_3, HeadPose& headPose);
	void headPoseEstimate(Mat& faceImg, const vector<Point2d>& facial5Pts, HeadPose& headPose);
	Mat headPoseEstimate(Mat& img, FaceInfo& faceInfo, HeadPose& headPose);
	vector<Mat> headPoseEstimate(Mat& img, vector<FaceInfo>& faceInfo, vector<HeadPose>& headPoses);
	void showheadPose(Mat& img, FaceInfo& faceInfo);
	void headPoseEstimation(Mat& img, FaceInfo& faceInfo);
}

#endif //__ZK_HEAD_POSE_ESTIMATION_H__
