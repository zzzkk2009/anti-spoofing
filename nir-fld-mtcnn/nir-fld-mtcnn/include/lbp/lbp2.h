#ifndef _LBP_2_H_
#define _LBP_2_H_

#include<opencv2/highgui/highgui.hpp>

using namespace cv;

Mat LBP(Mat img);
Mat ELBP(Mat img, int radius, int neighbors);
int getHopCount(uchar i);
Mat RILBP(Mat img);
Mat UniformLBP(Mat img);

#endif
