#ifndef __ZK_UTIL_H__
#define __ZK_UTIL_H__

#include <iostream>
#include "opencv2/opencv.hpp"
#include <numeric>
#include <io.h>
#include <direct.h>
#include <fstream>

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
}

#endif
