#ifndef __ZK_SVM_H__
#define __ZK_SVM_H__

#include <iostream>
#include "opencv2/opencv.hpp"
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>
#include <utils/util.h>
#include <string>

using namespace cv;
using namespace std;



namespace zk_svm {
	
	void train(string pos_fileName, string neg_fileName, string save_model_name);
	Ptr<cv::ml::SVM> load(string modelFile);
	float predict(Ptr<cv::ml::SVM> model, Mat data);
}

#endif
