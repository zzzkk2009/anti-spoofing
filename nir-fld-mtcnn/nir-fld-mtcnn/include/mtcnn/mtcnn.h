#ifndef __MTCNN_OPENCV_H__
#define __MTCNN_OPENCV_H__

//Created by Kai Zuo
#include <fstream>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <base/base.h>
using namespace std;
using namespace cv; 

//const float pnet_stride;
//const float pnet_cell_size;
//const int pnet_max_detect_num;
////mean & std
//const float mean_val;
//const float std_val;
////minibatch size
//const int step_size;


typedef struct FaceBox {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
} FaceBox;

//landmark [0:4]是五个关键点在水平方向上的坐标(列坐标)，landmark [5:9]是五个关键点在垂直方向上的坐标（行坐标）。
//即，（landmark [0]，landmark[1]）表示左眼的坐标，（landmark [2]，landmark[3]）表示右眼的坐标，（landmark [4]，landmark[5]）表示鼻子的坐标，
//（landmark [6]，landmark[7]）表示左嘴角的坐标，（landmark [8]，landmark[9]）表示右嘴角的坐标。
typedef struct FaceInfo {
	float bbox_reg[4];
	float landmark_reg[10];
	float landmark[10];
	FaceBox bbox;
} FaceInfo;
typedef struct FaceSize {
	float width;
	float height;
} FaceSize;

bool CompareBBox(const FaceInfo & a, const FaceInfo & b);

FaceInfo drawRectangle(Mat& img, vector<FaceInfo>& v);
FaceInfo_ncnn getMaxFaceInfo(vector<FaceInfo_ncnn>& v);
Rect FaceInfo2Rect(FaceInfo& faceInfo);
void cropFace4Flow(Mat& img, FaceInfo& faceInfo, Rect& cropInfo, int padding = 10);
void amendFaceAxis(Mat org_img, vector<FaceInfo>& facesInfo, int crop_width, int crop_height);
void amendFaceAxis(Mat org_img, vector<FaceInfo_ncnn>& facesInfo, int crop_width, int crop_height);
float faceInfoArea(FaceInfo& faceInfo);
FaceSize getFaceSize(FaceInfo& faceInfo);
void FaceInfoNcnn2FaceInfo(FaceInfo_ncnn& fi_ncnn, FaceInfo& fi);
void FaceInfoNcnn2FaceInfo(vector<FaceInfo_ncnn>& fis_ncnn, vector<FaceInfo>& fis);
void FaceInfo2FaceInfoNcnn(FaceInfo& fi, FaceInfo_ncnn& fi_ncnn);
void FaceInfo2FaceInfoNcnn(vector<FaceInfo>& fis, vector<FaceInfo_ncnn>& fis_ncnn);
bool isFacingCamera(FaceInfo& faceInfo);//是否面向摄像头(只检测正脸)

class MTCNN {
public:
	MTCNN(const string& proto_model_dir);
	vector<FaceInfo> Detect_mtcnn(const cv::Mat& img, const int min_size, const float* threshold, const float factor, const int stage);
	//protected:
	vector<FaceInfo> ProposalNet(const cv::Mat& img, int min_size, float threshold, float factor);
	vector<FaceInfo> NextStage(const cv::Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold);
	void BBoxRegression(vector<FaceInfo>& bboxes);
	void BBoxPadSquare(vector<FaceInfo>& bboxes, int width, int height);
	void BBoxPad(vector<FaceInfo>& bboxes, int width, int height);
	void GenerateBBox(Mat* confidence, Mat* reg_box, float scale, float thresh);
	std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
	float IoU(float xmin, float ymin, float xmax, float ymax, float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom = false);



	//    std::shared_ptr<dnn::Net> PNet_;
	//    std::shared_ptr<dnn::Net> ONet_;
	//    std::shared_ptr<dnn::Net> RNet_;
public:
	dnn::Net PNet_;
	dnn::Net RNet_;
	dnn::Net ONet_;

	std::vector<FaceInfo> candidate_boxes_;
	std::vector<FaceInfo> total_boxes_;
};

#endif //__MTCNN_OPENCV_H__
