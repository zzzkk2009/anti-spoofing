#ifndef __ZK_DECTOR_H__
#define __ZK_DECTOR_H__

#include <iostream>
#include "opencv2/opencv.hpp"
#include <mtcnn/mtcnn.h>
#include <flow/flow.h>
#include <utils/util.h>
#include <net/mx_shuffle_predict_2.h>

using namespace cv;
using namespace std;

namespace detect {

	class Detector
	{
	public:
		Detector();
		int detectSpoofing(vector<Mat>& rgb_cameraFrames, vector<Mat>& ir_cameraFrames);
		int detectSpoofing2(vector<Mat>& rgb_cameraFrames, vector<Mat>& ir_cameraFrames);
		vector<FaceInfo> detectFace(Mat& img);
		int predict(Mat& img);
		bool getStartSampling();
		void setStartSampling(bool startSampling);
		bool getSamplePositiveData();
		void setSamplePositiveData(bool samplePositiveData);
		Mat getShowImgRGB();
		Mat getShowImgIR();
		Mat getMotion2color();

	private:
		PredictorHandle predictor = 0;
		MTCNN faceDetector;
		Rect detectFaceArea;
		float mtcnn_factor = 0.709f;
		float mtcnn_threshold[3] = { 0.7f, 0.6f, 0.6f };
		int mtcnn_minSize = 72; //minSize对应最小人脸尺寸:~(w x h）;72->~50x70, 96->~58x80, 120->~80x120, 240->~160x200
		int predict_img_min_width = 48;
		int predict_img_min_height = 64;
		bool _startSampling = false;
		bool _samplePositiveData = true;
		Mat _showImg_rgb;
		Mat _showImg_ir;
		Mat _motion2color;
	};
	
}


#endif //__FLOW_H__
