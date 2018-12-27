#ifndef __ZK_UTIL_H__
#define __ZK_UTIL_H__

#define ACCESS _access
#define MKDIR(a) _mkdir((a))

#include <iostream>
#include "opencv2/opencv.hpp"
#include <numeric>
#include <io.h>
#include <direct.h>
#include <fstream>
#include<time.h>
#include<string.h>

using namespace cv;
using namespace std;

namespace util {
	double computeVariance(vector<double> v);
	double computeStdDev(vector<double> v);
	void writeFile(string filename, vector<float> v);
	ofstream getOfstream(string dirName, string fileName);
	void saveTrainingData(ofstream* ofs, vector<float> v);
	void preproccessTrainingData(string filename);
	int countFileLines(string fileName);
	/*static void _split(const std::string &s, char delim,
		std::vector<std::string> &elems);*/
	std::vector<std::string> split(const std::string &s, char delim);
	void cropArea4Flow(Mat& img, Mat& cropedArea);
	void cropArea4Flow(Mat& img, Rect& cropRect, Mat& cropedArea);
	void gatherDataSet(Mat& img, string dirPath);
	void cf_findFileFromDir(string mainDir, vector<string> &files);
	void cf_findFileFromDir2(string mainDir, vector<string> &files);
	void cf_findFileFromDir(string mainDir, vector<int> &files);
	void cf_findSubDirFromDir(string mainDir, vector<int> &dirs);
	string cf_NextdirName(string mainDir);
	string cf_NextFileName(string mainDir);
	template <typename T, typename S>
	T fill_cast(const S& v, const int width, const char c);

	void getFrames(VideoCapture& rgb_camera, VideoCapture& ir_camera, vector<Mat>& rgb_cameraFrames, vector<Mat>& ir_cameraFrames, int frame_num = 1, int margin_frame = 0);
	void getFrames(VideoCapture& rgb_camera, vector<Mat>& rgb_cameraFrames, int frame_num, int margin_frame);
	void getDetectFaceArea(Mat& img, Rect& rect);
	void getDetectFaceArea2(Mat& img, int r_width, int r_height, Rect& rect);
	bool isInside(Rect rect1, Rect rect2);
	float rectIOU(const cv::Rect& rectA, const cv::Rect& rectB);
	string getStrftime();
	string getStrftime(string format);
	void getFiles(string path, vector<string>& files);
	int CreatDir(char *pszDir);
	void drawMaskLayer(Mat& img);
}

#endif
