#include <detector/detector.h>

using namespace cv;
using namespace std;
using namespace flow;
using namespace shuffle;
using namespace util;

namespace detect 
{
	Detector::Detector():faceDetector("./model/mtcnn") {
		string prefix = "./model/mx-flow-shuffle/";
		string symbol_file = prefix + "checkpoint_48x64_1w5k/checkpoint-symbol.json";
		string params_file = prefix + "checkpoint_48x64_1w5k/checkpoint-9950.params";
		//string synset_file = prefix + "synset.txt";
		int status = initPredictor(predictor, symbol_file, params_file, predict_img_min_height, predict_img_min_width);
		// Load synsets
		//vector<string> synsets = loadSynsets(synset_file.c_str());
		
	}

	bool Detector::getStartSampling() {
		return _startSampling;
	}
	void Detector::setStartSampling(bool startSampling) {
		_startSampling = startSampling;
	}
	bool Detector::getSamplePositiveData() {
		return  _samplePositiveData;
	}
	void Detector::setSamplePositiveData(bool samplePositiveData) {
		_samplePositiveData = samplePositiveData;
	}

	vector<FaceInfo> Detector::detectFace(Mat& img) {
		vector<FaceInfo> facesInfo = faceDetector.Detect_mtcnn(img, mtcnn_minSize, mtcnn_threshold, mtcnn_factor, 3);
		return facesInfo;
	}

	int Detector::predict(Mat& img) {
		int iResponse = shuffle::predict(predictor, img);
		return iResponse;
	}

	Mat Detector::getShowImgRGB() {
		return _showImg_rgb;
	}
	Mat Detector::getShowImgIR() {
		return _showImg_ir;
	}

	Mat Detector::getMotion2color() {
		return _motion2color;
	}

	int Detector::detectSpoofing(vector<Mat>& rgb_cameraFrames, vector<Mat>& ir_cameraFrames) {

		cv::TickMeter tm;
		tm.reset();
		tm.start();

		Mat rgb_cameraFrame_0 = rgb_cameraFrames.at(0);
		Mat ir_cameraFrame_0 = ir_cameraFrames.at(0);

		_showImg_rgb = rgb_cameraFrame_0.clone();
		_showImg_ir = ir_cameraFrame_0.clone();

		string showCtrlSs = "startSampling:";
		stringstream startSampling_ss, samplePositiveData_ss;
		startSampling_ss << _startSampling;
		showCtrlSs += startSampling_ss.str();
		samplePositiveData_ss << _samplePositiveData;
		showCtrlSs += " ,samplePositiveData:" + samplePositiveData_ss.str();
		cv::putText(_showImg_rgb, showCtrlSs, cv::Point(20, _showImg_rgb.rows - 25), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

		if (detectFaceArea.empty())
		{
			util::getDetectFaceArea(rgb_cameraFrame_0, detectFaceArea);
		}
		cv::rectangle(_showImg_rgb, detectFaceArea, cv::Scalar(255, 0, 0), 1);
		cv::rectangle(_showImg_ir, detectFaceArea, cv::Scalar(255, 0, 0), 1);

		double t = (double)cv::getTickCount();

		Mat displayedFrame(rgb_cameraFrame_0.size(), CV_8UC3);

		vector<FaceInfo> facesInfo_rgb = detectFace(rgb_cameraFrame_0);
		vector<FaceInfo> facesInfo_ir = detectFace(ir_cameraFrame_0);
		//vector<FaceInfo> facesInfo_ir;

		std::cout << "detect time," << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s"
			<< std::endl;

		tm.stop();
		int fps = 1000.0 / tm.getTimeMilli();
		std::stringstream ss;
		ss << fps;
		cv::putText(_showImg_rgb, ss.str() + "FPS",
			cv::Point(20, 45), 4, 0.5, cv::Scalar(0, 0, 125));

		//只要有一个摄像头检测到人脸，就需要进行活体判断
		if (!facesInfo_rgb.empty() || !facesInfo_ir.empty())
		{
			if (!facesInfo_rgb.empty())
			{
				FaceInfo maxFaceInfo_rgb = drawRectangle(_showImg_rgb, facesInfo_rgb);
				Rect maxFace_rgb = FaceInfo2Rect(maxFaceInfo_rgb);
				//没在指定矩形框内不检测
				if (!util::isInside(maxFace_rgb, detectFaceArea))
					return -2;

				//电子屏幕攻击(也有可能是人脸检测算法在近红外上精度比较低)
				if (facesInfo_ir.empty())
				{
					cv::putText(_showImg_rgb, "spoofing(ir_empty)", cv::Point(280, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
					cout << "spoofing(ir_empty)" << endl;
					return 1;
				}

				FaceInfo maxFaceInfo_ir = drawRectangle(_showImg_ir, facesInfo_ir);
				Rect maxFace_ir = FaceInfo2Rect(maxFaceInfo_ir);

				FaceSize rgb_faceSize = getFaceSize(maxFaceInfo_rgb);
				//太小的人脸不检测
				if (rgb_faceSize.height < predict_img_min_height || rgb_faceSize.width < predict_img_min_width)
				{
					return -3;
				}

				//纠正两幅图像位置偏移bug
				maxFace_rgb.x += 20;
				maxFace_rgb.y += 10;
				cv::rectangle(_showImg_ir, maxFace_rgb, cv::Scalar(0, 0, 255), 1);
				float iou = rectIOU(maxFace_rgb, maxFace_ir);
				stringstream str_iou;
				str_iou << iou;
				cv::putText(_showImg_rgb, "IOU:" + str_iou.str(), cv::Point(550, 400), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
				if (iou < 0.7) //两幅图像待检测活体人脸位置不匹配
				{
					cv::putText(_showImg_rgb, "spoofing(iou)", cv::Point(280, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
					cout << "spoofing(iou)" << endl;
					return 2;
				}

				//稠密光流法
				if (rgb_cameraFrames.size() > 1)
				{
					Rect rgb_0_cropInfo;
					cropFace4Flow(rgb_cameraFrame_0, maxFaceInfo_rgb, rgb_0_cropInfo);
					Mat rgb_prev_crop_img = rgb_cameraFrame_0(rgb_0_cropInfo);

					resize(rgb_prev_crop_img, rgb_prev_crop_img, Size(predict_img_min_width, predict_img_min_height));
					Mat rgb_prev_crop_img_gray;
					cvtColor(rgb_prev_crop_img, rgb_prev_crop_img_gray, CV_BGR2GRAY);

					Mat rgb_flow, rgb_motion2color;
					for (int i = 1; i < rgb_cameraFrames.size(); i++)
					{
						Mat rgb_cur_crop_img = rgb_cameraFrames.at(i)(rgb_0_cropInfo);
						resize(rgb_cur_crop_img, rgb_cur_crop_img, Size(predict_img_min_width, predict_img_min_height));
						Mat rgb_cur_crop_img_gray;
						cvtColor(rgb_cur_crop_img, rgb_cur_crop_img_gray, CV_BGR2GRAY);

						calcOpticalFlowFarneback(rgb_prev_crop_img_gray, rgb_cur_crop_img_gray, rgb_flow, 0.5, 3, 15, 3, 5, 1.2, 0);

						if (!rgb_flow.empty())
						{
							motionToColor(rgb_flow, rgb_motion2color);
							motionToVectorField(rgb_prev_crop_img, rgb_flow);
							imshow("rgb_motion2color", rgb_motion2color);
							imshow("rgb_prev_crop_img", rgb_prev_crop_img);

							std::swap(rgb_prev_crop_img_gray, rgb_cur_crop_img_gray);

							int iResponse = predict(rgb_motion2color);

							if (1 == iResponse)
							{
								cv::putText(_showImg_rgb, "pass(flow)", cv::Point(280, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
								return 3;
							}
							else
							{
								cv::putText(_showImg_rgb, "spoofing(flow)", cv::Point(280, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
								cout << "spoofing(flow)" << endl;
								return 4;
							}

							if (_startSampling)
							{
								if (_samplePositiveData)
								{
									util::gatherDataSet(rgb_motion2color, "./data/1/");
								}
								else
								{
									util::gatherDataSet(rgb_motion2color, "./data/0/");
								}
							}
						}
					}
				}
			}
			else {//近红外检测到人脸，可见光未检测到人脸：可能是在黑暗条件
				return 0;
			}
		}
		else
		{// 近红外、可见光均未检测到人脸
			return -1;
		}
		
		/*cv::imshow("rgb_cameraFrame_0", _showImg_rgb);
		cv::imshow("ir_cameraFrame_0", _showImg_ir);*/
	}

	int Detector::detectSpoofing2(vector<Mat>& rgb_cameraFrames, vector<Mat>& ir_cameraFrames) {

		cv::TickMeter tm;
		tm.reset();
		tm.start();

		if (ir_cameraFrames.size() < 2) { //至少两帧
			return -4;
		}

		Mat rgb_cameraFrame_0 = rgb_cameraFrames.at(0);
		Mat ir_cameraFrame_0 = ir_cameraFrames.at(0);

		_showImg_rgb = rgb_cameraFrame_0.clone();
		_showImg_ir = ir_cameraFrame_0.clone();

		string showCtrlSs = "startSampling:";
		stringstream startSampling_ss, samplePositiveData_ss;
		startSampling_ss << _startSampling;
		showCtrlSs += startSampling_ss.str();
		samplePositiveData_ss << _samplePositiveData;
		showCtrlSs += " ,samplePositiveData:" + samplePositiveData_ss.str();
		cv::putText(_showImg_rgb, showCtrlSs, cv::Point(20, _showImg_rgb.rows - 25), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

		if (detectFaceArea.empty())
		{
			util::getDetectFaceArea(rgb_cameraFrame_0, detectFaceArea);
		}
		cv::rectangle(_showImg_rgb, detectFaceArea, cv::Scalar(255, 0, 0), 1);
		cv::rectangle(_showImg_ir, detectFaceArea, cv::Scalar(255, 0, 0), 1);

		double t = (double)cv::getTickCount();

		Mat displayedFrame(rgb_cameraFrame_0.size(), CV_8UC3);

		vector<FaceInfo> facesInfo_rgb = detectFace(rgb_cameraFrame_0);

		std::cout << "detect time," << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s"
			<< std::endl;

		tm.stop();
		int fps = 1000.0 / tm.getTimeMilli();
		std::stringstream ss;
		ss << fps;
		cv::putText(_showImg_rgb, ss.str() + "FPS",
			cv::Point(20, 45), 4, 0.5, cv::Scalar(0, 0, 125));

		//只要可见光摄像头检测到人脸，就进行活体判断
		if (!facesInfo_rgb.empty())
		{
			FaceInfo maxFaceInfo_rgb = drawRectangle(_showImg_rgb, facesInfo_rgb);
			Rect maxFace_rgb = FaceInfo2Rect(maxFaceInfo_rgb);
			//没在指定矩形框内不检测
			if (!util::isInside(maxFace_rgb, detectFaceArea))
				return -2;

			FaceSize rgb_faceSize = getFaceSize(maxFaceInfo_rgb);
			//太小的人脸不检测
			if (rgb_faceSize.height < predict_img_min_height || rgb_faceSize.width < predict_img_min_width)
			{
				return -3;
			}

			//纠正两幅图像位置偏移bug
			maxFace_rgb.x += 20;
			maxFace_rgb.y += 10;
			cv::rectangle(_showImg_ir, maxFace_rgb, cv::Scalar(0, 0, 255), 1);
			
			//稠密光流法
			Rect ir_0_cropInfo;
			//直接用rgb检测到的人脸，纠正位置后，去裁剪近红外上的人脸区域
			cropFace4Flow(ir_cameraFrame_0, maxFaceInfo_rgb, ir_0_cropInfo);
			Mat ir_prev_crop_img = ir_cameraFrame_0(ir_0_cropInfo);

			resize(ir_prev_crop_img, ir_prev_crop_img, Size(predict_img_min_width, predict_img_min_height));
			Mat ir_prev_crop_img_gray;
			cvtColor(ir_prev_crop_img, ir_prev_crop_img_gray, CV_BGR2GRAY);

			Mat ir_flow, ir_motion2color;
			for (int i = 1; i < ir_cameraFrames.size(); i++)
			{
				Mat ir_cur_crop_img = ir_cameraFrames.at(i)(ir_0_cropInfo);
				resize(ir_cur_crop_img, ir_cur_crop_img, Size(predict_img_min_width, predict_img_min_height));
				Mat ir_cur_crop_img_gray;
				cvtColor(ir_cur_crop_img, ir_cur_crop_img_gray, CV_BGR2GRAY);

				calcOpticalFlowFarneback(ir_prev_crop_img_gray, ir_cur_crop_img_gray, ir_flow, 0.5, 3, 15, 3, 5, 1.2, 0);

				if (!ir_flow.empty())
				{
					motionToColor(ir_flow, ir_motion2color);
					_motion2color = ir_motion2color;
					motionToVectorField(ir_prev_crop_img, ir_flow);
					imshow("ir_motion2color", ir_motion2color);
					imshow("ir_prev_crop_img", ir_prev_crop_img);

					std::swap(ir_prev_crop_img_gray, ir_cur_crop_img_gray);

					if (_startSampling)
					{
						if (_samplePositiveData)
						{
							util::gatherDataSet(ir_motion2color, "./data/48x64/1/");
						}
						else
						{
							util::gatherDataSet(ir_motion2color, "./data/48x64/0/");
						}
					}

					int iResponse = predict(ir_motion2color);

					if (1 == iResponse)
					{
						cv::putText(_showImg_rgb, "pass(flow)", cv::Point(280, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
						return 1;
					}
					else
					{
						cv::putText(_showImg_rgb, "spoofing(flow)", cv::Point(280, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
						cout << "spoofing(flow)" << endl;
						return 2;
					}
				}
				else {//无法获取光流图像
					return 0;
				}
			}
		}
		else
		{//未检测到人脸
			return -1;
		}
	}
}
