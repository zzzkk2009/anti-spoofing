#include <detector/detector.h>

using namespace cv;
using namespace std;
using namespace flow;
using namespace shuffle;
using namespace util;
using namespace hpe;

namespace detect
{
	Detector::Detector():faceDetector_ncnn("./model/insightface/insightface_ncnn")
		, faceDetector("./model/mtcnn")
		, arcFace("./model/insightface/insightface_ncnn")
	{
		string prefix = "./model/mx-flow-shuffle/shufflenet_v2/";
		string symbol_file = prefix + "checkpoint_48x64_5w_shufflenet_v2_2.0m-symbol.json";
		string params_file = prefix + "checkpoint_48x64_limitedarea_1.3w_shufflenet_v2_2.0m-0671.params";
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

	vector<FaceInfo_ncnn> Detector::detectFace(ncnn::Mat& ncnn_img) {
		vector<FaceInfo_ncnn> facesInfo = faceDetector_ncnn.Detect(ncnn_img);
		return facesInfo;
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
		stringstream startSampling_ss, samplePositiveData_ss, continuousDetectSpoofingNum_ss;
		startSampling_ss << _startSampling;
		showCtrlSs += startSampling_ss.str();
		samplePositiveData_ss << _samplePositiveData;
		showCtrlSs += " ,samplePositiveData:" + samplePositiveData_ss.str();
		continuousDetectSpoofingNum_ss << _continuousDetectSpoofingNum;
		showCtrlSs += " ,continuousDetectSpoofingNum:" + continuousDetectSpoofingNum_ss.str();
		cv::putText(_showImg_rgb, showCtrlSs, cv::Point(20, _showImg_rgb.rows - 25), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

		double t = (double)cv::getTickCount();

		Mat displayedFrame(rgb_cameraFrame_0.size(), CV_8UC3);

		Mat willDetectFaceArea_rgb = rgb_cameraFrame_0(detectFaceArea);
		Mat willDetectFaceArea_ir = ir_cameraFrame_0(detectFaceArea);

		ncnn::Mat willDetectFaceArea_rgb_ncnn = ncnn::Mat::from_pixels(willDetectFaceArea_rgb.data, ncnn::Mat::PIXEL_BGR, willDetectFaceArea_rgb.cols, willDetectFaceArea_rgb.rows);
		vector<FaceInfo> facesInfo_rgb = detectFace(willDetectFaceArea_rgb);
		vector<FaceInfo_ncnn> facesInfo_rgb_ncnn;
		FaceInfo2FaceInfoNcnn(facesInfo_rgb, facesInfo_rgb_ncnn);
		amendFaceAxis(rgb_cameraFrame_0, facesInfo_rgb, crop_width, crop_height);
		vector<FaceInfo_ncnn> facesInfo_ir_ncnn;
		vector<FaceInfo> facesInfo_ir;
		ncnn::Mat willDetectFaceArea_ir_ncnn = ncnn::Mat::from_pixels(willDetectFaceArea_ir.data, ncnn::Mat::PIXEL_BGR, willDetectFaceArea_ir.cols, willDetectFaceArea_ir.rows);
		if (_startSampling) {
			facesInfo_ir = facesInfo_rgb;
		}
		else {
			facesInfo_ir = detectFace(willDetectFaceArea_ir);
			FaceInfo2FaceInfoNcnn(facesInfo_ir, facesInfo_ir_ncnn);
			amendFaceAxis(ir_cameraFrame_0, facesInfo_ir, crop_width, crop_height);
		}

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

			if (!isFacingCamera(maxFaceInfo_rgb)) {
				cv::putText(_showImg_rgb, "look at camera!", cv::Point(230, 100), cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar(0, 0, 125), 1);
				cout << "look at camera!" << endl;
				return -2;
			}

			if (!_startSampling) {
				if (facesInfo_rgb.size() != facesInfo_ir.size()) {
					// 电子屏幕类攻击（也有可能是检测算法问题，但通过限制区域，检测精度应该是能保证两边一致的）
					//仿真度不高的面具，头盔等；
					cv::putText(_showImg_rgb, "spoofing(phone)!", cv::Point(200, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1);
					cout << "spoofing!" << endl;
					return 11;
				}
			}

			//太小的人脸不检测
			if (rgb_faceSize.height < predict_img_min_height || rgb_faceSize.width < predict_img_min_width
				|| ir_faceSize.height < predict_img_min_height || ir_faceSize.width < predict_img_min_width)
			{
				return -3;
			}

			/*HeadPose headPose;
			Mat faceImg = headPoseEstimate(_showImg_rgb, maxFaceInfo_rgb, headPose);
			int show_x = detectFaceArea.x + detectFaceArea.width;
			int show_y = detectFaceArea.y;
			char ch[20];
			sprintf(ch, "yaw:%0.2f", headPose.yaw);
			putText(_showImg_rgb, ch, Point(show_x, show_y), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 1);
			sprintf(ch, "pitch:%0.2f", headPose.pitch);
			putText(_showImg_rgb, ch, Point(show_x, show_y+40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 1);
			sprintf(ch, "roll:%0.2f", headPose.roll);
			putText(_showImg_rgb, ch, Point(show_x, show_y+80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 1);*/

			//headPoseEstimation(_showImg_rgb, maxFaceInfo_rgb);

			//if (_continuousDetectSpoofingNum > 2 * _continuousDetectSpoofingThreshold) {
			//	_continuousDetectSpoofingNum = 0;
			//}


			FaceInfo_ncnn maxFaceInfo_ncnn = getMaxFaceInfo(facesInfo_rgb_ncnn);
			ncnn::Mat det1 = preprocess(willDetectFaceArea_rgb_ncnn, maxFaceInfo_ncnn);

			//连续检测为非活体的次数超过阈值
			if (_continuousDetectSpoofingNum > _continuousDetectSpoofingThreshold) {
				double start = (double)getTickCount();
				vector<float> feature = arcFace.getFeature(det1);
				cout << "Extraction Time: " << (getTickCount() - start) / getTickFrequency() << "s" << std::endl;

				float similarity = calcSimilar(_cacheFeature, feature);
				/*imshow("_cacheImg", _cacheImg);
				imshow("willDetectFaceArea_rgb", willDetectFaceArea_rgb);*/
				std::stringstream ss;
				ss << similarity;
				cv::putText(_showImg_rgb, "similar:" + ss.str(),
					cv::Point(20, 105), 4, 0.5, cv::Scalar(0, 0, 125));
				if (similarity >= 0.8) {
					cv::putText(_showImg_rgb, "spoofing(sim)!", cv::Point(200, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1);
					cout << "spoofing(sim)!" << endl;
					return 13;
				}
				else {//换了一个人，需要重新检测
					/*_cacheFeature.clear();
					_cacheImg.release();*/
					_continuousDetectSpoofingNum = 0;
				}
			}
			//连续检测为活体的次数超过阈值
			else if (-_continuousDetectSpoofingNum > _continuousDetectPassingThreshold) {
				double start = (double)getTickCount();
				vector<float> feature = arcFace.getFeature(det1);
				cout << "Extraction Time(1): " << (getTickCount() - start) / getTickFrequency() << "s" << std::endl;

				float similarity = calcSimilar(_cacheFeature, feature);
				/*imshow("_cacheImg", _cacheImg);
				imshow("willDetectFaceArea_rgb", willDetectFaceArea_rgb);*/
				std::stringstream ss;
				ss << similarity;
				cv::putText(_showImg_rgb, "similar:" + ss.str(),
					cv::Point(20, 105), 4, 0.5, cv::Scalar(0, 0, 125));
				if (similarity >= 0.8) {
					cv::putText(_showImg_rgb, "pass(sim)", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 0), 1);
					cout << "pass" << endl;
					return 2;
				}
				else {//换了一个人，需要重新检测
					/*_cacheFeature.clear();
					_cacheImg.release();*/
					_continuousDetectSpoofingNum = 0;
				}
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
				if (_continuousDetectSpoofingNum > 0) {
					_continuousDetectSpoofingNum = 0;
				}

				if (0 == _continuousDetectSpoofingNum) {
					_cacheFeature = arcFace.getFeature(det1);
					_cacheImg = willDetectFaceArea_rgb;
				}
				_continuousDetectSpoofingNum--;
				cv::putText(_showImg_rgb, "pass", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 0), 1);
				return 1;
			}
			else if (-1 == iResponse) {//无光流图像，未检测到人脸(近红外未检测到人脸，就不做活体检测)
				/*_cacheFeature.clear();
				_cacheImg.release();*/
				_continuousDetectSpoofingNum = 0;
				return 0;
			}
			else
			{
				if (_continuousDetectSpoofingNum < 0) {
					_continuousDetectSpoofingNum = 0;
				}

				if (0 == _continuousDetectSpoofingNum) {
					_cacheFeature = arcFace.getFeature(det1);
					_cacheImg = willDetectFaceArea_rgb;
				}
				_continuousDetectSpoofingNum++;
				cv::putText(_showImg_rgb, "spoofing!", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1);
				cout << "spoofing!" << endl;
				return 12;
			}
		}
		{//未检测到人脸
			/*_cacheFeature.clear();
			_cacheImg.release();*/
			_continuousDetectSpoofingNum = 0;
			return -1;
		}
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

		ncnn::Mat willDetectFaceArea_rgb_ncnn = ncnn::Mat::from_pixels(willDetectFaceArea_rgb.data, ncnn::Mat::PIXEL_BGR, willDetectFaceArea_rgb.cols, willDetectFaceArea_rgb.rows);
		vector<FaceInfo_ncnn> facesInfo_rgb_ncnn = detectFace(willDetectFaceArea_rgb_ncnn);
		vector<FaceInfo> facesInfo_rgb;
		FaceInfoNcnn2FaceInfo(facesInfo_rgb_ncnn, facesInfo_rgb);
		amendFaceAxis(rgb_cameraFrame_0, facesInfo_rgb, crop_width, crop_height);
		vector<FaceInfo_ncnn> facesInfo_ir_ncnn;
		vector<FaceInfo> facesInfo_ir;
		ncnn::Mat willDetectFaceArea_ir_ncnn = ncnn::Mat::from_pixels(willDetectFaceArea_ir.data, ncnn::Mat::PIXEL_BGR, willDetectFaceArea_ir.cols, willDetectFaceArea_ir.rows);
		if (_startSampling) {
			facesInfo_ir = facesInfo_rgb;
		}
		else {
			facesInfo_ir_ncnn = detectFace(willDetectFaceArea_ir_ncnn);
			FaceInfoNcnn2FaceInfo(facesInfo_ir_ncnn, facesInfo_ir);
			amendFaceAxis(ir_cameraFrame_0, facesInfo_ir, crop_width, crop_height);
		}

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
					return 2;
				}
			}

			//太小的人脸不检测
			if (rgb_faceSize.height < predict_img_min_height || rgb_faceSize.width < predict_img_min_width
				|| ir_faceSize.height < predict_img_min_height || ir_faceSize.width < predict_img_min_width)
			{
				return -2;
			}

			//连续检测为非活体的次数超过阈值
			if (_continuousDetectSpoofingNum >= _continuousDetectSpoofingThreshold) {
				FaceInfo_ncnn maxFaceInfo_ncnn = getMaxFaceInfo(facesInfo_rgb_ncnn);
				ncnn::Mat det1 = preprocess(willDetectFaceArea_rgb_ncnn, maxFaceInfo_ncnn);
				if (_continuousDetectSpoofingNum == _continuousDetectSpoofingThreshold) {
					//保存人脸原始图和特征
					double start = (double)getTickCount();
					_cacheFeature = arcFace.getFeature(det1);
					_cacheImg = willDetectFaceArea_rgb;
					cout << "Extraction(1) Time: " << (getTickCount() - start) / getTickFrequency() << "s" << std::endl;
				}
				else {
					double start = (double)getTickCount();
					vector<float> feature = arcFace.getFeature(det1);
					cout << "Extraction Time: " << (getTickCount() - start) / getTickFrequency() << "s" << std::endl;

					float similarity = calcSimilar(_cacheFeature, feature);
					if (similarity >= 0.6) {
						cv::putText(_showImg_rgb, "spoofing!", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1);
						cout << "spoofing!" << endl;
						return 4;
					}
					else {//换了一个人，需要重新检测
						_continuousDetectSpoofingNum = 0;
					}
				}
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
				_continuousDetectSpoofingNum = 0;
				cv::putText(_showImg_rgb, "pass", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 0), 1);
				return 1;
			}
			else if (-1 == iResponse) {//无光流图像，未检测到人脸(近红外未检测到人脸，就不做活体检测)
				return 0;
			}
			else
			{
				_continuousDetectSpoofingNum++;
				cv::putText(_showImg_rgb, "spoofing!", cv::Point(280, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255), 1);
				cout << "spoofing!" << endl;
				return 3;
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

	int Detector::getContinuousDetectSpoofingThreshold() {
		return _continuousDetectSpoofingThreshold;
	}

	void Detector::setContinuousDetectSpoofingThreshold(int continuousDetectSpoofingThreshold) {
		_continuousDetectSpoofingThreshold = continuousDetectSpoofingThreshold;
	}

	int Detector::getContinuousDetectPassingThreshold() {
		return _continuousDetectPassingThreshold;
	}

	void Detector::setContinuousDetectPassingThreshold(int continuousDetectPassingThreshold) {
		_continuousDetectPassingThreshold = continuousDetectPassingThreshold;
	}
}


