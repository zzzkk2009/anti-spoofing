#ifndef __ZK_FLOW_H__
#define __ZK_FLOW_H__

#include <iostream>
#include "opencv2/opencv.hpp"
#include <lbp/lbp3.h>

using namespace cv;
using namespace std;

namespace flow {

	enum FLOW_HIST_TYPE {
		FLOW_HIST_TYPE_1 = 1,
		FLOW_HIST_TYPE_2 = 2,
		FLOW_HIST_TYPE_4 = 4,
		FLOW_HIST_TYPE_5 = 5,
		FLOW_HIST_TYPE_9 = 9,
		FLOW_HIST_TYPE_12= 12,
		FLOW_HIST_TYPE_18 = 18,
		FLOW_HIST_TYPE_20 = 20
	};

	enum SAMPLE_MARGIN {
		SAMPLE_MARGIN_1 = 1,
		SAMPLE_MARGIN_2 = 2,
		SAMPLE_MARGIN_3 = 3
	};

	void makecolorwheel(vector<Scalar> &colorwheel);
	bool isErrorFlow(Mat& flow);
	void motionToColor(Mat& flow, Mat &color);
	void drawArrow(cv::Mat& img, cv::Point& pStart, cv::Point& pEnd, int len, int alpha,
		cv::Scalar& color, int thickness = 1, int lintType = 8);
	void motionToVectorField(Mat& img, Mat& flow);
	vector<int> calcFlowAngleHist(Mat& flow, FLOW_HIST_TYPE flowHistType);
	vector<float> extractFlowAnglFeature(Mat& flow, SAMPLE_MARGIN sampleMargin);
	void computeLBPFeature(Mat& motion2color, Mat& featureMat);
	void computeLBPFeature(Mat& motion2color, vector<float>& feature);
}


#endif //__FLOW_H__