#ifndef MTCNN_NCNN_H
#define MTCNN_NCNN_H

#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include "net.h"
#include <base/base.h>

using namespace std;

namespace mtdet_ncnn {
	class MtcnnDetector_ncnn {
	public:
		MtcnnDetector_ncnn(string model_folder = ".");
		~MtcnnDetector_ncnn();
		vector<FaceInfo_ncnn> Detect(ncnn::Mat img);
	private:
		float minsize = 20;
		float threshold[3] = { 0.6f, 0.7f, 0.8f };
		float factor = 0.709f;
		const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
		const float norm_vals[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
		ncnn::Net Pnet;
		ncnn::Net Rnet;
		ncnn::Net Onet;
		ncnn::Net Lnet;
		vector<FaceInfo_ncnn> Pnet_Detect(ncnn::Mat img);
		vector<FaceInfo_ncnn> Rnet_Detect(ncnn::Mat img, vector<FaceInfo_ncnn> bboxs);
		vector<FaceInfo_ncnn> Onet_Detect(ncnn::Mat img, vector<FaceInfo_ncnn> bboxs);
		void Lnet_Detect(ncnn::Mat img, vector<FaceInfo_ncnn> &bboxs);
		vector<FaceInfo_ncnn> generateBbox(ncnn::Mat score, ncnn::Mat loc, float scale, float thresh);
		void doNms(vector<FaceInfo_ncnn> &bboxs, float nms_thresh, string mode);
		void refine(vector<FaceInfo_ncnn> &bboxs, int height, int width, bool flag = false);
	};
}


#endif
