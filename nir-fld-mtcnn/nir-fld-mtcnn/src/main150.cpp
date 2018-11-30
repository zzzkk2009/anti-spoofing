//
//#include <mtcnn/mtcnn.h>
//#include <flow/flow.h>
//#include <lbp/lbp3.h>
//#include <utils/util.h>
////#include <svm/svm.h>
//#include <net/mx_shuffle_predict_2.h>
//
//using namespace cv;
//using namespace std;
//using namespace flow;
//
//int main(int argc, char **argv)
//{
//	bool startSampling = false;
//	bool samplePositiveData = true;
//
//	//flow:SVM
//	/*ofstream pos_ofs = util::getOfstream("./data/", "positive_2.txt");
//	ofstream neg_ofs = util::getOfstream("./data/", "negative_2.txt");*/
//	//Ptr<cv::ml::SVM> svm_model = zk_svm::load("flow_12321.xml");
//
//	//flow:shufflenet
//	// Create predictor
//	string prefix = "./model/mx-flow-shuffle/";
//	string symbol_file = prefix + "checkpoint-symbol.json";
//	string params_file = prefix + "checkpoint-0060.params";
//	string synset_file = prefix + "synset.txt";
//	PredictorHandle predictor = 0;
//	int status = initPredictor(predictor, symbol_file, params_file);
//	// Load synsets
//	vector<string> synsets = loadSynsets(synset_file.c_str());
//
//	float factor = 0.709f;
//	float threshold[3] = { 0.7f, 0.6f, 0.6f };
//	int minSize = 72; // 96 
//	MTCNN detector("./model/mtcnn");
//
//	VideoCapture rgb_camera(0);
//	VideoCapture ir_camera(1);
//
//	cv::TickMeter tm;
//
//	Mat rgb_prevgray, rgb_gray, rgb_flow, rgb_motion2color;
//	Mat ir_prevgray, ir_gray, ir_flow, ir_motion2color;
//	Mat prevgray, cur_gray, flow, motion2color;
//
//	while (true)
//	{
//		Mat rgb_cameraFrame, ir_cameraFrame, org_rgb_cameraFrame, org_ir_cameraFrame;
//		rgb_camera >> rgb_cameraFrame;
//		ir_camera >> ir_cameraFrame;
//		org_rgb_cameraFrame = rgb_cameraFrame.clone();
//		org_ir_cameraFrame = ir_cameraFrame.clone();
//
//		if (rgb_cameraFrame.empty())
//		{
//			std::cerr << "rgb_cameraFrame empty!!!" << std::endl;
//			getchar();
//			exit(1);
//		}
//
//		if (ir_cameraFrame.empty())
//		{
//			std::cerr << "ir_cameraFrame empty!!!" << std::endl;
//			getchar();
//			exit(2);
//		}
//
//		tm.reset();
//		tm.start();
//
//		double t = (double)cv::getTickCount();
//
//		Mat displayedFrame(rgb_cameraFrame.size(), CV_8UC3);
//
//		vector<FaceInfo> facesInfo_rgb = detector.Detect_mtcnn(rgb_cameraFrame, minSize, threshold, factor, 3);
//		vector<FaceInfo> facesInfo_ir = detector.Detect_mtcnn(ir_cameraFrame, minSize, threshold, factor, 3);
//
//		std::cout << "detect time," << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s"
//			<< std::endl;
//
//		tm.stop();
//		int fps = 1000.0 / tm.getTimeMilli();
//		std::stringstream ss;
//		ss << fps;
//		cv::putText(rgb_cameraFrame, ss.str() + "FPS",
//			cv::Point(20, 45), 4, 0.5, cv::Scalar(0, 0, 125));
//
//		FaceInfo maxFaceInfo_rgb = drawRectangle(rgb_cameraFrame, facesInfo_rgb);
//		FaceInfo maxFaceInfo_ir = drawRectangle(ir_cameraFrame, facesInfo_ir);
//
//		if (facesInfo_rgb.size() > 0 && facesInfo_ir.size() > 0 && facesInfo_rgb.size() == facesInfo_ir.size())
//		{
//
//		}
//		else
//		{
//			//cv::putText(rgb_cameraFrame, "non-living", cv::Point(280, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
//			cout << "non-living" << endl;
//		}
//
//
//
//		//稠密光流法
//		if (maxFaceInfo_rgb.bbox.score > 0)
//		{
//			Mat clipFace_resize;
//			util::cropArea4Flow(org_ir_cameraFrame, clipFace_resize);
//			cvtColor(clipFace_resize, ir_gray, CV_BGR2GRAY);
//
//			/*Mat org_ir_resize, org_rgb_resize;
//			resize(org_ir_cameraFrame, org_ir_resize, Size(160, 160));
//			resize(org_rgb_cameraFrame, org_rgb_resize, Size(160, 160));
//			cvtColor(org_ir_resize, ir_gray, CV_BGR2GRAY);
//			cvtColor(org_rgb_resize, rgb_gray, CV_BGR2GRAY);*/
//
//			//imshow("org_ir_resize", org_ir_resize);
//			/*imshow("cur_gray", cur_gray);*/
//
//			if (ir_prevgray.data)
//			{
//				try
//				{
//					calcOpticalFlowFarneback(ir_prevgray, ir_gray, ir_flow, 0.5, 3, 15, 3, 5, 1.2, 0);
//					motionToColor(ir_flow, ir_motion2color);
//					motionToVectorField(clipFace_resize, ir_flow);
//					imshow("clipFace_resize", clipFace_resize);
//					imshow("ir_gray", ir_gray);
//					imshow("ir_flow", ir_motion2color);
//
//					/*calcOpticalFlowFarneback(prevgray, cur_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
//					motionToColor(flow, motion2color);
//					motionToVectorField(org_ir_resize, flow);
//					imshow("motion2color", motion2color);*/
//
//					//calcOpticalFlowFarneback(rgb_prevgray, rgb_gray, rgb_flow, 0.5, 3, 15, 3, 5, 1.2, 0);
//					//motionToColor(rgb_flow, rgb_motion2color);
//					//motionToVectorField(org_rgb_resize, rgb_flow);
//					//imshow("rgb_motion2color", rgb_motion2color);
//					//imshow("rgb_org_resize_arrow", org_rgb_resize);
//
//					//calcOpticalFlowFarneback(ir_prevgray, ir_gray, ir_flow, 0.5, 3, 15, 3, 5, 1.2, 0);
//					//motionToColor(ir_flow, ir_motion2color);
//					//motionToVectorField(org_ir_resize, ir_flow);
//					//imshow("ir_motion2color", ir_motion2color);
//					//imshow("ir_org_resize_arrow", org_ir_resize);
//
//					//Mat rgb_motion2color2Org;
//					//resize(rgb_motion2color, rgb_motion2color2Org, Size(640, 480));
//					////imshow("rgb_motion2color2Org", rgb_motion2color2Org);
//
//					//Rect rgb_face_rect = FaceInfo2Rect(maxFaceInfo_rgb);
//					//Mat rgb_face_flow = rgb_motion2color2Org(rgb_face_rect);
//					//resize(rgb_face_flow, rgb_face_flow, Size(32, 32));
//					//imshow("rgb_face_flow", rgb_face_flow);
//
//					//Mat ir_motion2color2Org;
//					//resize(ir_motion2color, ir_motion2color2Org, Size(640, 480));
//					//imshow("ir_motion2color2Org", ir_motion2color2Org);
//
//					/*Mat ir_face_flow;
//					if (maxFaceInfo_ir.bbox.score > 0)
//					{
//						Rect ir_face_rect = FaceInfo2Rect(maxFaceInfo_ir);
//						ir_face_flow = ir_motion2color2Org(ir_face_rect);
//						resize(ir_face_flow, ir_face_flow, Size(32, 32));
//						imshow("ir_face_flow", ir_face_flow);
//					}*/
//
//					/*Mat featureMat;
//					computeLBPFeature(ir_motion2color, featureMat);
//					vector<float> feature;
//					feature.assign((float*)featureMat.datastart, (float*)featureMat.dataend);*/
//
//
//
//					string label_synset_str = "";
//					int iResponse = predict(predictor, synsets, ir_motion2color, label_synset_str);
//
//					//float fResponse = zk_svm::predict(svm_model, featureMat);
//					//int iResponse = (int)fResponse;
//
//					if (startSampling)
//					{
//						if (samplePositiveData)
//						{
//							//if (!iResponse)
//							//{
//							//	//util::saveTrainingData(&pos_ofs, feature);
//							//	util::gatherDataSet(rgb_face_flow, "./data/1/");
//							//	if (!ir_face_flow.empty())
//							//	{
//							//		util::gatherDataSet(ir_face_flow, "./data/1/");
//							//	}
//							//}
//
//							//util::gatherDataSet(rgb_face_flow, "./data/1/");
//							/*if (!ir_face_flow.empty())
//							{
//								util::gatherDataSet(ir_face_flow, "./data/1/");
//							}*/
//						}
//						else
//						{
//							//if (iResponse)
//							//{
//							//	//util::saveTrainingData(&neg_ofs, feature);
//							//	util::gatherDataSet(rgb_face_flow, "./data/0/");
//							//	if (!ir_face_flow.empty())
//							//	{
//							//		util::gatherDataSet(ir_face_flow, "./data/1/");
//							//	}
//							//}
//
//							//util::gatherDataSet(rgb_face_flow, "./data/0/");
//							/*if (!ir_face_flow.empty())
//							{
//								util::gatherDataSet(ir_face_flow, "./data/1/");
//							}*/
//
//						}
//					}
//
//
//					stringstream str_fResponse;
//					str_fResponse << iResponse;
//					putText(rgb_cameraFrame, "fResponse:" + str_fResponse.str(), cv::Point(450, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
//
//					if (1 == iResponse)
//					{
//						putText(rgb_cameraFrame, "living(flow)", cv::Point(280, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
//					}
//					else
//					{
//						putText(rgb_cameraFrame, "non-living(flow)", cv::Point(280, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
//						cout << "non-living" << endl;
//					}
//				}
//				catch (Exception& e)
//				{
//					std::cout << e.what() << std::endl;
//				}
//			}
//		}
//
//
//
//		string showCtrlSs = "startSampling:";
//		stringstream startSampling_ss, samplePositiveData_ss;
//		startSampling_ss << startSampling;
//		showCtrlSs += startSampling_ss.str();
//		samplePositiveData_ss << samplePositiveData;
//		showCtrlSs += " ,samplePositiveData:" + samplePositiveData_ss.str();
//		cv::putText(rgb_cameraFrame, showCtrlSs, cv::Point(20, rgb_cameraFrame.rows - 25), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
//		cv::imshow("mtcnn_rgb_cameraFrame", rgb_cameraFrame);
//		cv::imshow("mtcnn_ir_cameraFrame", ir_cameraFrame);
//
//		int c = waitKey(1);
//		if (27 == c) // esc
//		{
//			break;
//		}
//		if (32 == c) //空格
//		{
//			startSampling = !startSampling;
//		}
//		if (char(c) == 'p') // samplePositiveData
//		{
//			samplePositiveData = !samplePositiveData;
//		}
//
//		//std::swap(prevgray, cur_gray);
//		std::swap(rgb_prevgray, rgb_gray);
//		std::swap(ir_prevgray, ir_gray);
//	}
//
//	/*pos_ofs.close();
//	neg_ofs.close();*/
//
//	return 0;
//}