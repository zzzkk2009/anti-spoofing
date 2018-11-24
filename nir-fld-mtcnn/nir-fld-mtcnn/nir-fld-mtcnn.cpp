
#include <mtcnn/mtcnn.h>
//#include <mtcnn-light/mtcnn.h>
#include <flow/flow.h>
//#include <lbp/lbp2.h>
//#include <lbp/LBP.hpp>
#include <lbp/lbp3.h>
//#include <numeric>
//#include <functional>
#include <utils/util.h>
#include <svm/svm.h>

using namespace cv;
using namespace std;
//using namespace lbp;
using namespace flow;

int main(int argc, char **argv)
{
	bool startSampling = false;
	bool samplePositiveData = true;
	ofstream pos_ofs = util::getOfstream("./data/", "positive_2.txt");
	ofstream neg_ofs = util::getOfstream("./data/", "negative_2.txt");

	Ptr<cv::ml::SVM> svm_model = zk_svm::load("flow_12321.xml");


	float factor = 0.709f;
	float threshold[3] = { 0.7f, 0.6f, 0.6f };
	int minSize = 96; // 96 
	MTCNN detector("./src/mtcnn/model");
	//MTCNN detector("E:/srcs/anti-spoofing/nir-fld-mtcnn/nir-fld-mtcnn/src/mtcnn/model");

	VideoCapture rgb_camera(0);
	VideoCapture ir_camera(1);
	//std::vector<FaceInfo> rgb_faces, ir_faces;

	cv::TickMeter tm;

	Mat rgb_prevgray, rgb_gray, rgb_flow, rgb_motion2color;
	Mat ir_prevgray, ir_gray, ir_flow, ir_motion2color;

	//Mat rgb_prevgray, rgb_gray, ir_prevgray, ir_gray;
	
	while (true)
	{
		try
		{
		Mat rgb_cameraFrame, ir_cameraFrame, org_rgb_cameraFrame, org_ir_cameraFrame;
		rgb_camera >> rgb_cameraFrame;
		//org_rgb_cameraFrame = rgb_cameraFrame.clone();
		ir_camera >> ir_cameraFrame;
		org_ir_cameraFrame = ir_cameraFrame.clone();

		if (rgb_cameraFrame.empty())
		{
			std::cerr << "rgb_cameraFrame empty!!!" << std::endl;
			getchar();
			exit(1);
		}

		if (ir_cameraFrame.empty())
		{
			std::cerr << "ir_cameraFrame empty!!!" << std::endl;
			getchar();
			exit(2);
		}

		/*cout << "ir_cameraFrame.channels: " << ir_cameraFrame.channels() << endl;
		cout << "rgb_cameraFrame.channels: " << rgb_cameraFrame.channels() << endl;*/

		tm.reset();
		tm.start();

		double t = (double)cv::getTickCount();

		Mat displayedFrame(rgb_cameraFrame.size(), CV_8UC3);

		vector<FaceInfo> facesInfo_rgb = detector.Detect_mtcnn(rgb_cameraFrame, minSize, threshold, factor, 3);
		vector<FaceInfo> facesInfo_ir = detector.Detect_mtcnn(ir_cameraFrame, minSize, threshold, factor, 3);

		std::cout << "detect time," << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s"
			<< std::endl;

		tm.stop();
		int fps = 1000.0 / tm.getTimeMilli();
		std::stringstream ss;
		ss << fps;
		cv::putText(rgb_cameraFrame, ss.str() + "FPS",
			cv::Point(20, 45), 4, 0.5, cv::Scalar(0, 0, 125));

		
		FaceInfo maxFaceInfo_rgb = FaceInfo{};
		for (int i = 0; i < facesInfo_rgb.size(); i++) {
			FaceInfo faceInfo_rgb = facesInfo_rgb[i];
			int x = (int)faceInfo_rgb.bbox.xmin;
			int y = (int)faceInfo_rgb.bbox.ymin;
			int w = (int)(faceInfo_rgb.bbox.xmax - faceInfo_rgb.bbox.xmin + 1);
			int h = (int)(faceInfo_rgb.bbox.ymax - faceInfo_rgb.bbox.ymin + 1);
			stringstream str_x, str_y, str_h, str_w;
			str_x << x;
			str_y << y;
			str_h << h;
			str_w << w;
			putText(rgb_cameraFrame, "x:"+ str_x.str()+",y:"+str_y.str()+",h:"+ str_h.str() + ",w:" + str_w.str(), cv::Point(x, y), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
			cv::rectangle(rgb_cameraFrame, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 1);

			int area = w * h;

			if (0 == i)
			{
				maxFaceInfo_rgb = faceInfo_rgb;
			}
			else
			{
				int max_w = (int)(maxFaceInfo_rgb.bbox.xmax - maxFaceInfo_rgb.bbox.xmin + 1);
				int max_h = (int)(maxFaceInfo_rgb.bbox.ymax - maxFaceInfo_rgb.bbox.ymin + 1);
				int max_area = max_w * max_h;
				if (area > max_area)
				{
					maxFaceInfo_rgb = faceInfo_rgb;
				}
			}
		}
		putText(rgb_cameraFrame, "maxFace", Point(maxFaceInfo_rgb.bbox.xmin, maxFaceInfo_rgb.bbox.ymin-20), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
		
		for (int i = 0; i < facesInfo_ir.size(); i++) {
			FaceInfo faceInfo_ir = facesInfo_ir[i];
			int x = (int)faceInfo_ir.bbox.xmin;
			int y = (int)faceInfo_ir.bbox.ymin;
			int w = (int)(faceInfo_ir.bbox.xmax - faceInfo_ir.bbox.xmin + 1);
			int h = (int)(faceInfo_ir.bbox.ymax - faceInfo_ir.bbox.ymin + 1);
			cv::rectangle(ir_cameraFrame, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 1);
		}

		if (facesInfo_rgb.size() > 0 && facesInfo_ir.size() > 0 && facesInfo_rgb.size() == facesInfo_ir.size())
		{
			putText(rgb_cameraFrame, "living", cv::Point(280, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
			cout << "living" << endl;


			
			
		}
		else
		{
			putText(rgb_cameraFrame, "non-living", cv::Point(280, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
			cout << "non-living" << endl;
		}
		
		//稠密光流法
		if (maxFaceInfo_rgb.bbox.score > 0)
		{
			/*int max_x = (int)maxFaceInfo_rgb.bbox.xmin;
			int max_y = (int)maxFaceInfo_rgb.bbox.ymin;
			int max_w = (int)(maxFaceInfo_rgb.bbox.xmax - maxFaceInfo_rgb.bbox.xmin + 1);
			int max_h = (int)(maxFaceInfo_rgb.bbox.ymax - maxFaceInfo_rgb.bbox.ymin + 1);

			int clip_x = max_x;
			if (max_x > 0)
			{
				clip_x = max_x * 0.5;
			}

			int clip_y = max_y;
			if (max_y > 0)
			{
				clip_y = max_y * 0.5;
			}

			int clip_w = max_w;
			if ((max_x + max_w) < rgb_cameraFrame.cols)
			{
				clip_w = max_w + (max_x - clip_x) + (rgb_cameraFrame.cols - (max_x + max_w)) * 0.5;
			}

			int clip_h = max_h;
			if ((max_y + max_h) < rgb_cameraFrame.rows)
			{
				clip_h = max_h + (max_y - clip_y) + (rgb_cameraFrame.rows - (max_h + max_h)) * 0.5;
			}*/

			int clip_x = org_ir_cameraFrame.cols * 0.25;
			int clip_y = org_ir_cameraFrame.rows * 0.25;
			int clip_w = org_ir_cameraFrame.cols * 0.65;
			int clip_h = org_ir_cameraFrame.rows * 0.65;

			Rect clipArea = Rect(clip_x, clip_y, clip_w, clip_h);
			Mat clipFace = org_ir_cameraFrame(clipArea);
			Mat clipFace_resize;
			resize(clipFace, clipFace_resize, Size(150, 150));
			cvtColor(clipFace_resize, ir_gray, CV_BGR2GRAY);

			/*Mat rgb_flow_frame;
			resize(org_rgb_cameraFrame, rgb_flow_frame, Size(210, 120));
			cvtColor(rgb_flow_frame, rgb_gray, CV_BGR2GRAY);*/

			if (ir_prevgray.data)
			{
				try
				{
					calcOpticalFlowFarneback(ir_prevgray, ir_gray, ir_flow, 0.5, 3, 15, 3, 5, 1.2, 0);
					motionToColor(ir_flow, ir_motion2color);
					motionToVectorField(clipFace_resize, ir_flow);
					//imshow("rgb_clipFace", clipFace);
					imshow("clipFace_resize", clipFace_resize);
					imshow("ir_gray", ir_gray);
					imshow("ir_flow", ir_motion2color);

					/*vector<float> feature = extractFlowAnglFeature(ir_flow, SAMPLE_MARGIN_2);*/

					/*vector<int> flowHist = calcFlowAngleHist(ir_flow, FLOW_HIST_TYPE_1);
					std::vector<float> feature;
					for (int i = 0; i < flowHist.size(); i++)
					{
						feature.push_back(float(flowHist[i]));
					}*/

					Mat featureMat, ir_motion2color_gray;
					cvtColor(ir_motion2color, ir_motion2color_gray, CV_BGR2GRAY);
					UniformRotInvLBPFeature(ir_motion2color_gray, Size(4, 4), featureMat);
					featureMat.convertTo(featureMat, CV_32F);
					vector<float> feature;
					feature.assign((float*)featureMat.datastart, (float*)featureMat.dataend);
					
					if (startSampling)
					{
						if (samplePositiveData)
						{
							util::saveTrainingData(&pos_ofs, feature);
						}
						else
						{
							util::saveTrainingData(&neg_ofs, feature);
						}
					}

					//Mat featureMat = Mat::zeros(1, feature.size(), CV_32FC1);
					//memcpy(featureMat.data, feature.data(), feature.size() * sizeof(float));
					
					float fResponse = zk_svm::predict(svm_model, featureMat);
					stringstream str_fResponse;
					str_fResponse << fResponse;
					putText(rgb_cameraFrame, "fResponse:" + str_fResponse.str(), cv::Point(400, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
					

					//util::writeFile("feature.txt", feature);

					//vector<int> flowHist = calcFlowAngleHist(ir_flow, FLOW_HIST_TYPE_1);

					//if (flowHist.size() > 0)
					//{
					//	double sum = accumulate(std::begin(flowHist), std::end(flowHist), 0.0);
					//	double mean = sum / flowHist.size(); //均值

					//	double accum = 0.0;
					//	std::for_each(std::begin(flowHist), std::end(flowHist), [&](const double d) {
					//		accum += (d - mean)*(d - mean);
					//	});

					//	double stdev = sqrt(accum / (flowHist.size() - 1)); //方差
					//	stringstream str_stdev;
					//	str_stdev << stdev;
					//	putText(rgb_cameraFrame, "stdev:" + str_stdev.str(), cv::Point(360, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
					//}
				}
				catch (Exception& e)
				{
					std::cout << e.what() << std::endl;
				}
				
				
				


				//ofstream ofs;
				//ofs.open("hist_1.txt", ios::app); // ios::out ios::app
				//ofs << endl;

				//// Print out the histogram values
				//double sum = 0;
				//cout << "flowHist = [";
				//for (int i = 0; i < flowHist.size(); i++) {
				//	cout << flowHist[i] << ", ";
				//	sum += flowHist[i];

				//	if (i > 0) ofs << ", ";
				//	ofs << flowHist[i];
				//}
				//cout << "]; " << endl;
				//cout << "flowHist sum=" << sum << endl;

				//	
				//ofs << endl;
				//ofs << "flowHist sum=" << sum;
				//ofs << endl;
				//ofs.close();




				/*Mat rgb_lbp = LBP(rgb_motion2color);
				Mat rgb_elbp = ELBP(rgb_motion2color, 1, 8);
				Mat rgb_rilbp = RILBP(rgb_motion2color);
				Mat rgb_uniformlbp = UniformLBP(rgb_motion2color);
				imshow("rgb_lbp", rgb_lbp);
				imshow("rgb_elbp", rgb_elbp);
				imshow("rgb_rilbp", rgb_rilbp);
				imshow("rgb_uniformlbp", rgb_uniformlbp);*/

				/*Mat rgb_motion2color_gray;
				cvtColor(rgb_motion2color, rgb_motion2color_gray, CV_BGR2GRAY);
				Mat rgb_gray_lbp = LBP(rgb_motion2color_gray);
				imshow("rgb_motion2color_gray", rgb_motion2color_gray);*/


				//------LBP.cpp--------------
				//int rad = 1;
				//int pts = 8;
				//bool outputHist = true, normalizeHist = false;
				//Mat rgb_motion2color_gray;
				//cvtColor(rgb_motion2color, rgb_motion2color_gray, CV_BGR2GRAY);
				//// convert to double precision
				//rgb_motion2color_gray.convertTo(rgb_motion2color_gray, CV_64F);

				//Mat lbpImg;
				//lbpImg = Mat(rgb_motion2color_gray.size(), CV_8UC1, Scalar(0));

				//// Create an LBP instance of type "mapping" using "pts" support points
				//LBP lbp(pts, LBP_MAPPING_HF);

				//for (int i = 0; i < rgb_motion2color_gray.channels(); i++) {
				//	// Copy channel i
				//	Mat img(rgb_motion2color_gray.size(), rgb_motion2color_gray.depth(), 1);
				//	const int from_to1[] = { i, 0 };
				//	mixChannels(&rgb_motion2color_gray, 1, &img, 1, from_to1, 1);

				//	// Calculate the descriptor
				//	lbp.calcLBP(img, rad, true);

				//	// Copy lbp image
				//	const int from_to2[] = { 0, i };
				//	Mat tmpImg = lbp.getLBPImage();
				//	mixChannels(&tmpImg, 1, &lbpImg, 1, from_to2, 1);
				//}

				//if (outputHist) {
				//	// Calculate Fourier tranformed histogram
				//	vector<double> hist = lbp.calcHist().getHist(normalizeHist);

				//	ofstream ofs;
				//	ofs.open("hist_1.txt", ios::app); // ios::out ios::app
				//	ofs << endl;

				//	// Print out the histogram values
				//	double sum = 0;
				//	cout << "hist = [";
				//	for (int i = 0; i < hist.size(); i++) {
				//		cout << hist[i] << ", ";
				//		sum += hist[i];

				//		if (i > 0) ofs << ", ";
				//		ofs << hist[i];
				//	}
				//	cout << "]; " << endl;
				//	cout << "hist sum=" << sum << endl;

				//	
				//	ofs << endl;
				//	ofs << "hist sum=" << sum;
				//	ofs << endl;
				//	ofs.close();
				//}
				//------LBP.cpp--------------


			}
		}
		

		/*if (ir_prevgray.data)
		{
			calcOpticalFlowFarneback(ir_prevgray, ir_gray, ir_flow, 0.5, 3, 15, 3, 5, 1.2, 0);
			motionToColor(ir_flow, ir_motion2color);
			imshow("ir_flow", ir_motion2color);
		}*/




		//稀疏光流法
		//vector<Point2f> rgb_features , ir_features;
		//vector<Point2f> rgb_features_after, ir_features_after;
		//vector<uchar> rgb_status, ir_status;
		//vector<float> rgb_err, ir_err;
		//int maxCout = 300;//定义最大个数
		//double minDis = 10;//定义最小距离
		//double qLevel = 0.01;//定义质量水平
		
		//cvtColor(rgb_cameraFrame, rgb_gray, CV_BGR2GRAY);
		//cvtColor(ir_cameraFrame, ir_gray, CV_BGR2GRAY);
		//
		//if (rgb_prevgray.data)
		//{
		//	//检测第一帧的特征点
		//	goodFeaturesToTrack(rgb_prevgray, rgb_features, maxCout, qLevel, minDis);
		//	//计算出第二帧的特征点
		//	calcOpticalFlowPyrLK(rgb_prevgray, rgb_gray, rgb_features, rgb_features_after, rgb_status, rgb_err);
		//	//判别哪些属于运动的特征点
		//	int k = 0;
		//	for (int i = 0; i < rgb_features_after.size(); i++)
		//	{
		//		//状态要是1，并且坐标要移动下的那些点
		//		if (rgb_status[i] && ((abs(rgb_features[i].x - rgb_features_after[i].x) +
		//			abs(rgb_features[i].y - rgb_features_after[i].y)) > 4))
		//		{
		//			rgb_features_after[k++] = rgb_features_after[i];
		//		}
		//	}
		//	rgb_features_after.resize(k);//截取
		//	cout << k << endl;
		//	for (int i = 0; i < rgb_features_after.size(); i++)
		//	{
		//		//将特征点画一个小圆出来--粗细为2
		//		circle(rgb_prevgray, rgb_features_after[i], 3, Scalar(255), 2);
		//	}
		//	imshow("rgb_flow_sparse", rgb_prevgray);
		//}

		//if (ir_prevgray.data)
		//{
		//	//检测第一帧的特征点
		//	goodFeaturesToTrack(ir_prevgray, ir_features, maxCout, qLevel, minDis);
		//	//计算出第二帧的特征点
		//	calcOpticalFlowPyrLK(ir_prevgray, ir_gray, ir_features, ir_features_after, ir_status, ir_err);
		//	//判别哪些属于运动的特征点
		//	int k = 0;
		//	for (int i = 0; i < ir_features_after.size(); i++)
		//	{
		//		//状态要是1，并且坐标要移动下的那些点
		//		if (ir_status[i] && ((abs(ir_features[i].x - ir_features_after[i].x) +
		//			abs(ir_features[i].y - ir_features_after[i].y)) > 4))
		//		{
		//			ir_features_after[k++] = ir_features_after[i];
		//		}
		//	}
		//	ir_features_after.resize(k);//截取
		//	cout << k << endl;
		//	for (int i = 0; i < ir_features_after.size(); i++)
		//	{
		//		//将特征点画一个小圆出来--粗细为2
		//		circle(ir_prevgray, ir_features_after[i], 3, Scalar(255), 2);
		//	}
		//	imshow("flow_ir_cameraFrame", ir_prevgray);
		//}
		
		string showCtrlSs = "startSampling:";
		stringstream startSampling_ss, samplePositiveData_ss;
		startSampling_ss << startSampling;
		showCtrlSs += startSampling_ss.str();
		samplePositiveData_ss << samplePositiveData;
		showCtrlSs += " ,samplePositiveData:" + samplePositiveData_ss.str();
		putText(rgb_cameraFrame, showCtrlSs, cv::Point(20, rgb_cameraFrame.rows - 25), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
		cv::imshow("mtcnn_rgb_cameraFrame", rgb_cameraFrame);
		cv::imshow("mtcnn_ir_cameraFrame", ir_cameraFrame);


		int c = waitKey(1);
		if (27 == c) // esc
		{
			break;
		}
		if (32 == c) //空格
		{
			startSampling = !startSampling;
		}
		if (char(c) == 'p') // samplePositiveData
		{
			samplePositiveData = !samplePositiveData;
		}

		//std::swap(rgb_prevgray, rgb_gray);
		std::swap(ir_prevgray, ir_gray);
		}
		catch (Exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}

	pos_ofs.close();
	neg_ofs.close();

	return 0;
}