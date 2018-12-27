#include <iostream>
#include "opencv2/opencv.hpp"
#include <utils/util.h>
using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
	VideoCapture rgb_camera(0);
	VideoCapture ir_camera(1);
	float _frameRate = 25.0;//视频的帧率
	Size _videoSize = Size(640, 480);
	int frameCount = 0;
	VideoWriter _vWriter_rgb;
	VideoWriter _vWriter_nir;
	string date = util::getStrftime("%Y%m%d%H");

	while (true)
	{
		try
		{
			
			Mat rgb_cameraFrame_0, nir_cameraFrame_0;
			rgb_camera >> rgb_cameraFrame_0;
			ir_camera >> nir_cameraFrame_0;

			int crop_width = 230;
			int crop_height = 230;
			Rect detectFaceArea;
			util::getDetectFaceArea2(rgb_cameraFrame_0, crop_width, crop_height, detectFaceArea);
			Mat roi_rgb = rgb_cameraFrame_0(detectFaceArea).clone();
			util::drawMaskLayer(rgb_cameraFrame_0);
			roi_rgb.copyTo(rgb_cameraFrame_0(detectFaceArea));

			if (frameCount >= _frameRate * 10) { // 每个视频10秒
				frameCount = 0;
			}

			if (0 == frameCount) {
				string rgb_hourPath = "./videos/limitedArea/" + date + "/0/rgbs/";
				util::CreatDir((char *)rgb_hourPath.c_str());
				string nir_hourPath = "./videos/limitedArea/" + date + "/0/nirs/";
				util::CreatDir((char *)nir_hourPath.c_str());
				string _strftime = util::getStrftime();
				_vWriter_rgb = VideoWriter(rgb_hourPath + _strftime + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), _frameRate, _videoSize);
				_vWriter_nir = VideoWriter(nir_hourPath + _strftime + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), _frameRate, _videoSize);
			}

			frameCount++;
			_vWriter_rgb << rgb_cameraFrame_0;
			_vWriter_nir << nir_cameraFrame_0;

			cv::imshow("showImg_rgb", rgb_cameraFrame_0);
			cv::imshow("showImg_ir", nir_cameraFrame_0);

			int c = waitKey(1);
			if (27 == c) // esc
			{
				break;
			}
		}
		catch (Exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}

}


//int main(int argc, char **argv)
//{
//	VideoCapture rgb_camera(0);
//	VideoCapture ir_camera(1);
//	float _frameRate = 25.0;//视频的帧率
//	Size _videoSize = Size(640, 480);
//	int _positiveFrameCount = 0;
//	int _negativeFrameCount = 0;
//	VideoWriter _vWriter_rgb_p;
//	VideoWriter _vWriter_nir_p;
//	VideoWriter _vWriter_rgb_n;
//	VideoWriter _vWriter_nir_n;
//	bool _startSampling = false;
//	bool _samplePositiveData = true;
//	Rect detectFaceArea;
//
//	while (true)
//	{
//		try
//		{
//			cv::TickMeter tm;
//			tm.reset();
//			tm.start();
//
//			Mat rgb_cameraFrame_0, nir_cameraFrame_0;
//			rgb_camera >> rgb_cameraFrame_0;
//			ir_camera >> nir_cameraFrame_0;
//
//			Mat showImg_rgb = rgb_cameraFrame_0.clone();
//			Mat showImg_ir = nir_cameraFrame_0.clone();
//
//			tm.stop();
//			int fps = 1000.0 / tm.getTimeMilli();
//			std::stringstream ss;
//			ss << fps;
//			cv::putText(showImg_rgb, ss.str() + "FPS",
//				cv::Point(20, 45), 4, 0.5, cv::Scalar(0, 0, 125));
//
//			if (_startSampling)
//			{
//				if (_samplePositiveData) {
//					if (_positiveFrameCount >= _frameRate * 10) {
//						_positiveFrameCount = 0;
//					}
//
//					if (0 == _positiveFrameCount) {
//						string _strftime = util::getStrftime();
//						_vWriter_rgb_p = VideoWriter("./videos/1/rgbs/" + _strftime + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), _frameRate, _videoSize);
//						_vWriter_nir_p = VideoWriter("./videos/1/nirs/" + _strftime + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), _frameRate, _videoSize);
//					}
//
//					_positiveFrameCount++;
//					_vWriter_rgb_p << rgb_cameraFrame_0;
//					_vWriter_nir_p << nir_cameraFrame_0;
//				}
//				else {
//					if (_negativeFrameCount >= _frameRate * 10) {
//						_negativeFrameCount = 0;
//					}
//
//					if (0 == _negativeFrameCount) {
//						string _strftime = util::getStrftime();
//						_vWriter_rgb_n = VideoWriter("./videos/0/rgbs/" + _strftime + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), _frameRate, _videoSize);
//						_vWriter_nir_n = VideoWriter("./videos/0/nirs/" + _strftime + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), _frameRate, _videoSize);
//					}
//
//					_negativeFrameCount++;
//					_vWriter_rgb_n << rgb_cameraFrame_0;
//					_vWriter_nir_n << nir_cameraFrame_0;
//				}
//			}
//
//			if (detectFaceArea.empty())
//			{
//				util::getDetectFaceArea(rgb_cameraFrame_0, detectFaceArea);
//			}
//			cv::rectangle(showImg_rgb, detectFaceArea, cv::Scalar(255, 0, 0), 1);
//
//			string showCtrlSs = "startSampling:";
//			stringstream startSampling_ss, samplePositiveData_ss;
//			startSampling_ss << _startSampling;
//			showCtrlSs += startSampling_ss.str();
//			samplePositiveData_ss << _samplePositiveData;
//			showCtrlSs += " ,samplePositiveData:" + samplePositiveData_ss.str();
//			cv::putText(showImg_rgb, showCtrlSs, cv::Point(20, showImg_rgb.rows - 25), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
//
//			cv::imshow("showImg_rgb", showImg_rgb);
//			cv::imshow("showImg_ir", showImg_ir);
//
//			int c = waitKey(1);
//			if (27 == c) // esc
//			{
//				break;
//			}
//			if (32 == c) //空格
//			{
//				_startSampling = !_startSampling;
//			}
//			if (char(c) == 'p') // samplePositiveData
//			{
//				_samplePositiveData = !_samplePositiveData;
//			}
//		}
//		catch (Exception& e)
//		{
//			std::cout << e.what() << std::endl;
//		}
//	}
//
//}