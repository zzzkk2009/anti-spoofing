#ifndef _LBP_3_H_
#define _LBP_3_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>  
//using namespace std;
#include <iostream>  
#include <fstream>  
#include <sstream>  

using namespace cv;
using namespace std;

uchar GetMinBinary(uchar *binary);
int ComputeValue9(int value58);
void NormalLBPImage(const Mat &srcImage, Mat &LBPImage);
void UniformNormalLBPImage(const Mat &srcImage, Mat &LBPImage);
void UniformRotInvLBPImage(const Mat &srcImage, Mat &LBPImage);
void NormalLBPFeature(const Mat &srcImage, Size cellSize, Mat &featureVector);
void UniformNormalLBPFeature(const Mat &srcImage, Size cellSize, Mat &featureVector);
void UniformRotInvLBPFeature(const Mat &srcImage, Size cellSize, Mat &featureVector);


#endif