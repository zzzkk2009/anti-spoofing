#include <detector/detector.h>

using namespace cv;
using namespace std;
using namespace flow;
using namespace shuffle;
using namespace util;

namespace detect 
{
	Detector::Detector():faceDetector("./model/mtcnn") {
		string prefix = "./model/mx-flow-shuffle/shufflenet_v2/";
		string symbol_file = prefix + "checkpoint_48x64_5w_shufflenet_v2_2.0m-symbol.json";
		string params_file = prefix + "checkpoint_48x64_5w_shufflenet_v2_2.0m-0240.params";
		//string synset_file = prefix + "synset.txt";
		int status = initPredictor(predictor, symbol_file, params_file, predict_img_min_height, predict_img_min_width);
		// Load synsets
		//vector<string> synsets = loadSynsets(synset_file.c_str());
		
	}

	Detector::Detector(int method):faceDetector("./model/mtcnn") {
		string prefix = "./model/mx-flow-shuffle/shufflenet_v2/";
		if (1 == method) {
			string symbol_file_rgb = prefix + "checkpoint_48x64_rgbs_good_10w_shufflenet_v2_2.0m-symbol.json";
			string params_file_rgb = prefix + "checkpoint_48x64_rgbs_good_10w_shufflenet_v2_2.0m-0410.params";
			int status = initPredictor(predictor_rgb, symbol_file_rgb, params_file_rgb, predict_img_min_height, predict_img_min_width);
		}
		string symbol_file = prefix + "checkpoint_48x64_5w_shufflenet_v2_2.0m-symbol.json";
		string params_file = prefix + "checkpoint_48x64_30w_shufflenet_v2_2.0m-0320.params";
		int status = initPredictor(predictor, symbol_file, params_file, predict_img_min_height, predict_img_min_width);
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

	int Detector::predict(PredictorHandle _predictor, Mat& img) {
		int iResponse = shuffle::predict(_predictor, img);
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

	void Detector::gatherVideoDataSet(Mat& rgb_img, Mat& nir_img) {
		if (_startSampling)
		{
			if (_samplePositiveData)
			{
				if (_positiveFrameCount >= _frameRate * 10) {
					_positiveFrameCount = 0;
					/*_vWriter_rgb.release();
					_vWriter_nir.release();*/
				}

				if (0 == _positiveFrameCount) {
					string _strftime = getStrftime();
					_vWriter_rgb_p = VideoWriter("./videos/1/rgbs/" + _strftime + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), _frameRate, _videoSize);
					_vWriter_nir_p = VideoWriter("./videos/1/nirs/" + _strftime + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), _frameRate, _videoSize);
				}

				_positiveFrameCount++;
				_vWriter_rgb_p << rgb_img;
				_vWriter_nir_p << nir_img;
				
			}
			else
			{
				if (_negativeFrameCount >= _frameRate * 10) {
					_negativeFrameCount = 0;
					/*_vWriter_rgb.release();
					_vWriter_nir.release();*/
				}

				if (0 == _negativeFrameCount) {
					string _strftime = getStrftime();
					_vWriter_rgb_n = VideoWriter("./videos/0/rgbs/" + _strftime + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), _frameRate, _videoSize);
					_vWriter_nir_n = VideoWriter("./videos/0/nirs/" + _strftime + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), _frameRate, _videoSize);
				}

				_negativeFrameCount++;
				_vWriter_rgb_n << rgb_img;
				_vWriter_nir_n << nir_img;
			}

			
		}
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
					cv::cvtColor(rgb_prev_crop_img, rgb_prev_crop_img_gray, CV_BGR2GRAY);

					Mat rgb_flow, rgb_motion2color;
					for (int i = 1; i < rgb_cameraFrames.size(); i++)
					{
						Mat rgb_cur_crop_img = rgb_cameraFrames.at(i)(rgb_0_cropInfo);
						resize(rgb_cur_crop_img, rgb_cur_crop_img, Size(predict_img_min_width, predict_img_min_height));
						Mat rgb_cur_crop_img_gray;
						cv::cvtColor(rgb_cur_crop_img, rgb_cur_crop_img_gray, CV_BGR2GRAY);

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
			/*if (!util::isInside(maxFace_rgb, detectFaceArea))
				return -2;*/

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
			cv::cvtColor(ir_prev_crop_img, ir_prev_crop_img_gray, CV_BGR2GRAY);

			Mat ir_flow, ir_motion2color;
			for (int i = 1; i < ir_cameraFrames.size(); i++)
			{
				Mat ir_cur_crop_img = ir_cameraFrames.at(i)(ir_0_cropInfo);
				resize(ir_cur_crop_img, ir_cur_crop_img, Size(predict_img_min_width, predict_img_min_height));
				Mat ir_cur_crop_img_gray;
				cv::cvtColor(ir_cur_crop_img, ir_cur_crop_img_gray, CV_BGR2GRAY);

				calcOpticalFlowFarneback(ir_prev_crop_img_gray, ir_cur_crop_img_gray, ir_flow, 0.5, 3, 15, 3, 5, 1.2, 0);

				if (!ir_flow.empty())
				{
					motionToColor(ir_flow, ir_motion2color);
					_motion2color = ir_motion2color;
					//motionToVectorField(ir_prev_crop_img, ir_flow);
					imshow("ir_motion2color", ir_motion2color);
					//imshow("ir_prev_crop_img", ir_prev_crop_img);

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
						cv::putText(_showImg_rgb, "pass", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 0), 1);
						return 1;
					}
					else
					{
						cv::putText(_showImg_rgb, "spoofing!", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1);
						cout << "spoofing!" << endl;
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

	int Detector::detectSpoofing3(vector<Mat>& rgb_cameraFrames, vector<Mat>& ir_cameraFrames) {

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
			/*if (!util::isInside(maxFace_rgb, detectFaceArea))
				return -2;*/

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

			vector<Mat> rgb_motion2colors, ir_motion2colors;
			int type = 0;
			getFlowMotion2Colors(rgb_cameraFrames, ir_cameraFrames, type, maxFaceInfo_rgb, maxFaceInfo_rgb, rgb_motion2colors, ir_motion2colors);
			if (1 == type) {
				for (int i = 0; i < ir_motion2colors.size(); i++) {
					Mat ir_motion2color = ir_motion2colors.at(i);
					Mat rgb_motion2color = rgb_motion2colors.at(i);
					imshow("ir_motion2color_" + i, ir_motion2color);
					imshow("rgb_motion2color_" + i, rgb_motion2color);
					gatherFlowDataSet(ir_motion2color, 1);
					gatherFlowDataSet(rgb_motion2color, 0);
				}
			}
			/*else if (2 == type) {
				for (int i = 0; i < rgb_motion2colors.size(); i++) {
					Mat rgb_motion2color = rgb_motion2colors.at(i);
					imshow("rgb_motion2color_" + i, rgb_motion2color);
					gatherFlowDataSet(rgb_motion2color, 0);
				}
			}*/
			else {
				for (int i = 0; i < ir_motion2colors.size(); i++) {
					Mat ir_motion2color = ir_motion2colors.at(i);
					imshow("ir_motion2color_" + i, ir_motion2color);
					gatherFlowDataSet(ir_motion2color, 1);
				}
			}

			int iResponse = predict(ir_motion2colors);
			if (1 == iResponse)
			{
				cv::putText(_showImg_rgb, "pass", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 0), 1);
				return 1;
			}
			else if (-1 == iResponse) {//无光流图像，未检测到人脸
				return 0;
			}
			else
			{
				cv::putText(_showImg_rgb, "spoofing!", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1);
				cout << "spoofing!" << endl;
				return 2;
			}
		}
		else
		{//未检测到人脸
			return -1;
		}
	}

	int Detector::detectSpoofing4(vector<Mat>& rgb_cameraFrames) {

		cv::TickMeter tm;
		tm.reset();
		tm.start();

		if (rgb_cameraFrames.size() < 2) { //至少两帧
			return -4;
		}

		Mat rgb_cameraFrame_0 = rgb_cameraFrames.at(0);

		_showImg_rgb = rgb_cameraFrame_0.clone();

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
			/*if (!util::isInside(maxFace_rgb, detectFaceArea))
				return -2;*/

			FaceSize rgb_faceSize = getFaceSize(maxFaceInfo_rgb);
			//太小的人脸不检测
			if (rgb_faceSize.height < predict_img_min_height || rgb_faceSize.width < predict_img_min_width)
			{
				return -3;
			}

			
			vector<Mat> rgb_motion2colors, ir_motion2colors;
			int type = 2;
			getFlowMotion2Colors(rgb_cameraFrames, rgb_cameraFrames, type, maxFaceInfo_rgb, maxFaceInfo_rgb, rgb_motion2colors, rgb_motion2colors);

			if (1 == type) {
				for (int i = 0; i < ir_motion2colors.size(); i++) {
					Mat ir_motion2color = ir_motion2colors.at(i);
					Mat rgb_motion2color = rgb_motion2colors.at(i);
					imshow("ir_motion2color_" + i, ir_motion2color);
					imshow("rgb_motion2color_" + i, rgb_motion2color);
					gatherFlowDataSet(ir_motion2color, 1);
					gatherFlowDataSet(rgb_motion2color, 0);
				}
			}
			else if (2 == type) {
				for (int i = 0; i < rgb_motion2colors.size(); i++) {
					Mat rgb_motion2color = rgb_motion2colors.at(i);
					imshow("rgb_motion2color_" + i, rgb_motion2color);
					gatherFlowDataSet(rgb_motion2color, 0);
				}
			}
			else {
				for (int i = 0; i < ir_motion2colors.size(); i++) {
					Mat ir_motion2color = ir_motion2colors.at(i);
					imshow("ir_motion2color_" + i, ir_motion2color);
					gatherFlowDataSet(ir_motion2color, 1);
				}
			}

			int iResponse;
			if (2 == type) {
				iResponse = predict(rgb_motion2colors);
			}
			else {
				iResponse = predict(ir_motion2colors);
			}

			if (1 == iResponse)
			{
				cv::putText(_showImg_rgb, "pass", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 0), 1);
				return 1;
			}
			else if (-1 == iResponse) {//无光流图像，未检测到人脸
				return 0;
			}
			else
			{
				cv::putText(_showImg_rgb, "spoofing!", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1);
				cout << "spoofing!" << endl;
				return 2;
			}
		}
		else
		{//未检测到人脸
			return -1;
		}
	}

	int Detector::detectSpoofing5(vector<Mat>& rgb_cameraFrames, vector<Mat>& ir_cameraFrames) {

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
			/*if (!util::isInside(maxFace_rgb, detectFaceArea))
				return -2;*/

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

			vector<Mat> rgb_motion2colors, ir_motion2colors;
			int type = 1;
			getFlowMotion2Colors(rgb_cameraFrames, ir_cameraFrames, type, maxFaceInfo_rgb, maxFaceInfo_rgb, rgb_motion2colors, ir_motion2colors);
			if (1 == type) {
				for (int i = 0; i < ir_motion2colors.size(); i++) {
					Mat ir_motion2color = ir_motion2colors.at(i);
					Mat rgb_motion2color = rgb_motion2colors.at(i);
					imshow("ir_motion2color_" + i, ir_motion2color);
					imshow("rgb_motion2color_" + i, rgb_motion2color);
					gatherFlowDataSet(ir_motion2color, 1);
					gatherFlowDataSet(rgb_motion2color, 0);
				}
			}
			/*else if (2 == type) {
				for (int i = 0; i < rgb_motion2colors.size(); i++) {
					Mat rgb_motion2color = rgb_motion2colors.at(i);
					imshow("rgb_motion2color_" + i, rgb_motion2color);
					gatherFlowDataSet(rgb_motion2color, 0);
				}
			}*/
			else {
				for (int i = 0; i < ir_motion2colors.size(); i++) {
					Mat ir_motion2color = ir_motion2colors.at(i);
					imshow("ir_motion2color_" + i, ir_motion2color);
					gatherFlowDataSet(ir_motion2color, 1);
				}
			}

			int iResponse = predict(ir_motion2colors);
			int iResponse_rgb = predict(predictor_rgb, rgb_motion2colors);
			
			if (1 == iResponse && 1 == iResponse_rgb)
			{
				cv::putText(_showImg_rgb, "pass", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 0), 1);
				return 1;
			}
			else if (-1 == iResponse) {//无光流图像，未检测到人脸(近红外未检测到人脸，就不做活体检测)
				return 0;
			}
			else
			{
				cv::putText(_showImg_rgb, "spoofing!", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1);
				cout << "spoofing!" << endl;
				return 2;
			}
		}
		else
		{//未检测到人脸
			return -1;
		}
	}

	int Detector::detectSpoofing6(vector<Mat>& rgb_cameraFrames, vector<Mat>& ir_cameraFrames) {

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

		int crop_width = 230;
		int crop_height = 230;
		if (detectFaceArea.empty())
		{
			util::getDetectFaceArea2(rgb_cameraFrame_0, crop_width, crop_height, detectFaceArea);
		}
		/*cv::rectangle(_showImg_rgb, detectFaceArea, cv::Scalar(255, 0, 0), 1);
		cv::rectangle(_showImg_ir, detectFaceArea, cv::Scalar(255, 0, 0), 1);*/
		Mat roi_rgb = _showImg_rgb(detectFaceArea).clone();
		util::drawMaskLayer(_showImg_rgb);
		roi_rgb.copyTo(_showImg_rgb(detectFaceArea));

		string showCtrlSs = "startSampling:";
		stringstream startSampling_ss, samplePositiveData_ss;
		startSampling_ss << _startSampling;
		showCtrlSs += startSampling_ss.str();
		samplePositiveData_ss << _samplePositiveData;
		showCtrlSs += " ,samplePositiveData:" + samplePositiveData_ss.str();
		cv::putText(_showImg_rgb, showCtrlSs, cv::Point(20, _showImg_rgb.rows - 25), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

		double t = (double)cv::getTickCount();

		Mat displayedFrame(rgb_cameraFrame_0.size(), CV_8UC3);

		Mat willDetectFaceArea_rgb = rgb_cameraFrame_0(detectFaceArea);
		Mat willDetectFaceArea_ir = ir_cameraFrame_0(detectFaceArea);
		vector<FaceInfo> facesInfo_rgb = detectFace(willDetectFaceArea_rgb);
		vector<FaceInfo> facesInfo_ir;
		if (_startSampling) {
			facesInfo_ir = facesInfo_rgb;
		}
		else {
			facesInfo_ir = detectFace(willDetectFaceArea_ir);
		}

		amendFaceAxis(rgb_cameraFrame_0, facesInfo_rgb, crop_width, crop_height);
		amendFaceAxis(ir_cameraFrame_0, facesInfo_ir, crop_width, crop_height);

		std::cout << "detect time," << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s"
			<< std::endl;

		tm.stop();
		int fps = 1000.0 / tm.getTimeMilli();
		std::stringstream ss;
		ss << fps;
		cv::putText(_showImg_rgb, ss.str() + "FPS",
			cv::Point(20, 45), 4, 0.5, cv::Scalar(0, 0, 125));

		if (!facesInfo_rgb.empty()) {
			//两幅图像检测到的人脸数一致，进行下一步光流检测
			FaceInfo maxFaceInfo_rgb = drawRectangle(_showImg_rgb, facesInfo_rgb);
			FaceSize rgb_faceSize = getFaceSize(maxFaceInfo_rgb);
			FaceInfo maxFaceInfo_ir;
			if (_startSampling) {
				Rect maxFace_rgb = FaceInfo2Rect(maxFaceInfo_rgb);
				//纠正两幅图像位置偏移bug
				maxFace_rgb.x += 20;
				maxFace_rgb.y += 10;
				cv::rectangle(_showImg_ir, maxFace_rgb, cv::Scalar(0, 0, 255), 1);
				maxFaceInfo_ir = maxFaceInfo_rgb;
			}
			else {
				maxFaceInfo_ir = drawRectangle(_showImg_ir, facesInfo_ir);
			}
			FaceSize ir_faceSize = getFaceSize(maxFaceInfo_ir);

			if (!_startSampling) {
				if (facesInfo_rgb.size() != facesInfo_ir.size()) {
					// 电子屏幕类攻击（也有可能是检测算法问题，但通过限制区域，检测精度应该是能保证两边一致的）
					//仿真度不高的面具，头盔等；
					cv::putText(_showImg_rgb, "spoofing!", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1);
					cout << "spoofing!" << endl;
					return -2;
				}
			}

			//太小的人脸不检测
			if (rgb_faceSize.height < predict_img_min_height || rgb_faceSize.width < predict_img_min_width
				|| ir_faceSize.height < predict_img_min_height || ir_faceSize.width < predict_img_min_width)
			{
				return -3;
			}

			vector<Mat> rgb_motion2colors, ir_motion2colors;
			int type = 0;
			getFlowMotion2Colors(rgb_cameraFrames, ir_cameraFrames, type, maxFaceInfo_rgb, maxFaceInfo_ir, rgb_motion2colors, ir_motion2colors);

			if (1 == type) {
				for (int i = 0; i < ir_motion2colors.size(); i++) {
					Mat ir_motion2color = ir_motion2colors.at(i);
					Mat rgb_motion2color = rgb_motion2colors.at(i);
					imshow("ir_motion2color_" + i, ir_motion2color);
					imshow("rgb_motion2color_" + i, rgb_motion2color);
					gatherFlowDataSet(ir_motion2color, 1);
					gatherFlowDataSet(rgb_motion2color, 0);
				}
			}
			else {
				for (int i = 0; i < ir_motion2colors.size(); i++) {
					Mat ir_motion2color = ir_motion2colors.at(i);
					imshow("ir_motion2color_" + i, ir_motion2color);
					gatherFlowDataSet(ir_motion2color, 1);
				}
			}

			int iResponse = predict(ir_motion2colors);

			if (1 == iResponse)
			{
				cv::putText(_showImg_rgb, "pass", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 0), 1);
				return 1;
			}
			else if (-1 == iResponse) {//无光流图像，未检测到人脸(近红外未检测到人脸，就不做活体检测)
				return 0;
			}
			else
			{
				cv::putText(_showImg_rgb, "spoofing!", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1);
				cout << "spoofing!" << endl;
				return 2;
			}

		}
		{//未检测到人脸
			return -1;
		}
	}

	int Detector::predict(vector<Mat>& motion2colors) {
		int size = motion2colors.size();
		if (0 == size)
			return -1;
		int correctNum = 0;
		for (int i = 0; i < size; i++) {
			int iResponse = predict(motion2colors.at(i));
			if (1 == iResponse) {
				correctNum++;
			}
		}
		float percentage = correctNum / size;
		if (percentage > 0.5) {
			return 1;
		}
		else {
			return 0;
		}
	}

	int Detector::predict(PredictorHandle _predictor, vector<Mat>& motion2colors) {
		int size = motion2colors.size();
		if (0 == size)
			return -1;
		int correctNum = 0;
		for (int i = 0; i < size; i++) {
			int iResponse = predict(_predictor, motion2colors.at(i));
			if (1 == iResponse) {
				correctNum++;
			}
		}
		float percentage = correctNum / size;
		if (percentage > 0.5) {
			return 1;
		}
		else {
			return 0;
		}
	}

	void Detector::gatherFlowDataSet(Mat& motion2color, int type) {
		if (_startSampling)
		{

			if (_samplePositiveData)
			{
				if (1 == type) {
					util::gatherDataSet(motion2color, "./data/48x64/limitedArea/nirs/1/");
				}
				else {
					util::gatherDataSet(motion2color, "./data/48x64/rgbs/1/");
					//util::gatherDataSet(motion2color, "./data/48x64/ai_challenger_short_video/group7-good/1/");
				}
			}
			else
			{
				if (1 == type) {
					util::gatherDataSet(motion2color, "./data/48x64/limitedArea/nirs/0/");
				}
				else {
					util::gatherDataSet(motion2color, "./data/48x64/rgbs/0/");
				}
			}
		}
	}

	void Detector::getFlowMotion2Colors(vector<Mat>& rgb_cameraFrames, vector<Mat>& ir_cameraFrames, int type, FaceInfo maxFaceInfo_rgb, FaceInfo maxFaceInfo_ir, vector<Mat>& rgb_motion2colors, vector<Mat>& ir_motion2colors)
	{
		//稠密光流法
		if (rgb_cameraFrames.size() >= 2 || ir_cameraFrames.size() >= 2)
		{
			Mat rgb_cameraFrame_0 = rgb_cameraFrames.at(0);
			Mat ir_cameraFrame_0 = ir_cameraFrames.at(0);
			Rect rgb_0_cropInfo, ir_0_cropInfo;
			Mat ir_prev_crop_img, rgb_prev_crop_img;
			Mat ir_prev_crop_img_gray, rgb_prev_crop_img_gray;

			Mat ir_flow, rgb_flow, ir_motion2color, rgb_motion2color;

			if (1 == type) {
				//直接用rgb检测到的人脸，纠正位置后，去裁剪近红外上的人脸区域
				cropFace4Flow(ir_cameraFrame_0, maxFaceInfo_ir, ir_0_cropInfo);
				ir_prev_crop_img = ir_cameraFrame_0(ir_0_cropInfo);
				resize(ir_prev_crop_img, ir_prev_crop_img, Size(predict_img_min_width, predict_img_min_height));
				cvtColor(ir_prev_crop_img, ir_prev_crop_img_gray, CV_BGR2GRAY);

				cropFace4Flow(rgb_cameraFrame_0, maxFaceInfo_rgb, rgb_0_cropInfo);
				rgb_prev_crop_img = rgb_cameraFrame_0(rgb_0_cropInfo);
				resize(rgb_prev_crop_img, rgb_prev_crop_img, Size(predict_img_min_width, predict_img_min_height));
				cvtColor(rgb_prev_crop_img, rgb_prev_crop_img_gray, CV_BGR2GRAY);

				for (int i = 1; i < ir_cameraFrames.size(); i++)
				{
					Mat ir_cur_crop_img = ir_cameraFrames.at(i)(ir_0_cropInfo);
					resize(ir_cur_crop_img, ir_cur_crop_img, Size(predict_img_min_width, predict_img_min_height));
					Mat ir_cur_crop_img_gray;
					cvtColor(ir_cur_crop_img, ir_cur_crop_img_gray, CV_BGR2GRAY);

					calcOpticalFlowFarneback(ir_prev_crop_img_gray, ir_cur_crop_img_gray, ir_flow, 0.5, 3, 15, 3, 5, 1.2, 0);

					if (!ir_flow.empty())
					{
						if (!isErrorFlow(ir_flow)) {
							motionToColor(ir_flow, ir_motion2color);
							_motion2color = ir_motion2color;
							ir_motion2colors.push_back(ir_motion2color);
						}
						std::swap(ir_prev_crop_img_gray, ir_cur_crop_img_gray);
					}

					Mat rgb_cur_crop_img = rgb_cameraFrames.at(i)(rgb_0_cropInfo);
					resize(rgb_cur_crop_img, rgb_cur_crop_img, Size(predict_img_min_width, predict_img_min_height));
					Mat rgb_cur_crop_img_gray;
					cvtColor(rgb_cur_crop_img, rgb_cur_crop_img_gray, CV_BGR2GRAY);
					calcOpticalFlowFarneback(rgb_prev_crop_img_gray, rgb_cur_crop_img_gray, rgb_flow, 0.5, 3, 15, 3, 5, 1.2, 0);
					if (!rgb_flow.empty())
					{
						if (!isErrorFlow(rgb_flow)) {
							motionToColor(rgb_flow, rgb_motion2color);
							rgb_motion2colors.push_back(rgb_motion2color);
						}
						std::swap(rgb_prev_crop_img_gray, rgb_cur_crop_img_gray);
					}
				}
			}
			else if (2 == type) {
				cropFace4Flow(rgb_cameraFrame_0, maxFaceInfo_rgb, rgb_0_cropInfo);
				rgb_prev_crop_img = rgb_cameraFrame_0(rgb_0_cropInfo);
				resize(rgb_prev_crop_img, rgb_prev_crop_img, Size(predict_img_min_width, predict_img_min_height));
				cvtColor(rgb_prev_crop_img, rgb_prev_crop_img_gray, CV_BGR2GRAY);

				for (int i = 1; i < rgb_cameraFrames.size(); i++)
				{
					Mat rgb_cur_crop_img = rgb_cameraFrames.at(i)(rgb_0_cropInfo);
					resize(rgb_cur_crop_img, rgb_cur_crop_img, Size(predict_img_min_width, predict_img_min_height));
					Mat rgb_cur_crop_img_gray;
					cvtColor(rgb_cur_crop_img, rgb_cur_crop_img_gray, CV_BGR2GRAY);
					calcOpticalFlowFarneback(rgb_prev_crop_img_gray, rgb_cur_crop_img_gray, rgb_flow, 0.5, 3, 15, 3, 5, 1.2, 0);
					if (!rgb_flow.empty())
					{
						if (!isErrorFlow(rgb_flow)) {
							motionToColor(rgb_flow, rgb_motion2color);
							rgb_motion2colors.push_back(rgb_motion2color);
						}
						std::swap(rgb_prev_crop_img_gray, rgb_cur_crop_img_gray);
					}
				}
			}
			else {
				if (_startSampling) {
					cropFace4Flow(ir_cameraFrame_0, maxFaceInfo_rgb, ir_0_cropInfo);
				}
				else {
					cropFace4Flow(ir_cameraFrame_0, maxFaceInfo_ir, ir_0_cropInfo);
				}
				ir_prev_crop_img = ir_cameraFrame_0(ir_0_cropInfo);
				resize(ir_prev_crop_img, ir_prev_crop_img, Size(predict_img_min_width, predict_img_min_height));
				cvtColor(ir_prev_crop_img, ir_prev_crop_img_gray, CV_BGR2GRAY);
				for (int i = 1; i < ir_cameraFrames.size(); i++)
				{
					Mat ir_cur_crop_img = ir_cameraFrames.at(i)(ir_0_cropInfo);
					resize(ir_cur_crop_img, ir_cur_crop_img, Size(predict_img_min_width, predict_img_min_height));
					Mat ir_cur_crop_img_gray;
					cvtColor(ir_cur_crop_img, ir_cur_crop_img_gray, CV_BGR2GRAY);
					calcOpticalFlowFarneback(ir_prev_crop_img_gray, ir_cur_crop_img_gray, ir_flow, 0.5, 3, 15, 3, 5, 1.2, 0);
					if (!ir_flow.empty())
					{
						if (!isErrorFlow(ir_flow)) {
							motionToColor(ir_flow, ir_motion2color);
							_motion2color = ir_motion2color;
							//motionToVectorField(ir_prev_crop_img, ir_flow);
							//imshow("ir_motion2color", ir_motion2color);
							//imshow("ir_prev_crop_img", ir_prev_crop_img);
							ir_motion2colors.push_back(ir_motion2color);
						}
						std::swap(ir_prev_crop_img_gray, ir_cur_crop_img_gray);
					}
				}
			}
		}
	}
}


