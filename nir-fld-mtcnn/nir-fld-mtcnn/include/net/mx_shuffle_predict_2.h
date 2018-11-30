#ifndef __ZK_MX_SHUFFLE_PREDICT_2_H__
#define __ZK_MX_SHUFFLE_PREDICT_2_H__

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "mxnet/c_predict_api.h"

using namespace std;
using namespace cv;

string readAllBytes(const char *filename);
int initPredictor(PredictorHandle &predictor, string symbol_file, string params_file, mx_uint img_height, mx_uint img_width);
Mat preprocess(const cv::Mat& img, int num_channels, cv::Size input_geometry);
vector<string> loadSynsets(const char *filename);
int predict(PredictorHandle predictor, vector<string>& synsets, Mat& img, string label_synset_str, mx_uint img_height, mx_uint img_width);

#endif
