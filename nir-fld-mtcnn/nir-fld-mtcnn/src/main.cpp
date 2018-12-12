
#include <mtcnn/mtcnn.h>
#include <flow/flow.h>
#include <utils/util.h>
#include <net/mx_shuffle_predict_2.h>

using namespace cv;
using namespace std;
using namespace flow;
using namespace shuffle;

int main(int argc, char **argv)
{
	bool startSampling = false;
	bool samplePositiveData = true;
	bool is_train = false;

	//flow:shufflenet
	//Create predictor
	string prefix = "./model/mx-flow-shuffle/";
	string symbol_file = prefix + "checkpoint_c48x64_1w5k/checkpoint_c48x64_1w5k-symbol.json";
	string params_file = prefix + "checkpoint_c48x64_1w5k/checkpoint_c48x64_1w5k-30000.params";
	string synset_file = prefix + "synset.txt";
	PredictorHandle predictor = 0;
	vector<int> train_img_heights = {64}; // {32, 64}
	vector<int> train_img_widths = { 48 }; // { 32, 48 }
	mx_uint predict_img_height = 64, predict_img_width = 48;
	int status = initPredictor(predictor, symbol_file, params_file, predict_img_height, predict_img_width);
	// Load synsets
	//vector<string> synsets = loadSynsets(synset_file.c_str());

	float factor = 0.709f;
	float threshold[3] = { 0.7f, 0.6f, 0.6f };
	int minSize = 72; //minSize对应最小人脸尺寸:~(w x h）;72->~50x70, 96->~58x80, 120->~80x120, 240->~160x200
	MTCNN detector("./model/mtcnn");

	VideoCapture rgb_camera(0);
	VideoCapture ir_camera(1);

	cv::TickMeter tm;

	Rect detectFaceArea;

	try
	{
		while (true)
		{
			vector<Mat> rgb_cameraFrames, ir_cameraFrames;
			util::getFrames(rgb_camera, ir_camera, rgb_cameraFrames, ir_cameraFrames, 2, 0);

			Mat rgb_cameraFrame_0 = rgb_cameraFrames.at(0);
			Mat ir_cameraFrame_0 = ir_cameraFrames.at(0);

			Mat org_rgb_cameraFrame_0 = rgb_cameraFrame_0.clone();
			Mat org_ir_cameraFrame_0 = ir_cameraFrame_0.clone();

			if (detectFaceArea.empty())
			{
				util::getDetectFaceArea(org_rgb_cameraFrame_0, detectFaceArea);
			}
			cv::rectangle(org_rgb_cameraFrame_0, detectFaceArea, cv::Scalar(255, 0, 0), 1);
			cv::rectangle(org_ir_cameraFrame_0, detectFaceArea, cv::Scalar(255, 0, 0), 1);

			tm.reset();
			tm.start();

			double t = (double)cv::getTickCount();

			Mat displayedFrame(rgb_cameraFrame_0.size(), CV_8UC3);

			vector<FaceInfo> facesInfo_rgb = detector.Detect_mtcnn(rgb_cameraFrame_0, minSize, threshold, factor, 3);
			vector<FaceInfo> facesInfo_ir = detector.Detect_mtcnn(ir_cameraFrame_0, minSize, threshold, factor, 3);
			//vector<FaceInfo> facesInfo_ir;

			std::cout << "detect time," << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s"
				<< std::endl;

			tm.stop();
			int fps = 1000.0 / tm.getTimeMilli();
			std::stringstream ss;
			ss << fps;
			cv::putText(org_rgb_cameraFrame_0, ss.str() + "FPS",
				cv::Point(20, 45), 4, 0.5, cv::Scalar(0, 0, 125));

			//只要有一个摄像头检测到人脸，就需要进行活体判断
			if (!facesInfo_rgb.empty() || !facesInfo_ir.empty())
			{
				if (!facesInfo_rgb.empty())
				{
					FaceInfo maxFaceInfo_rgb = drawRectangle(org_rgb_cameraFrame_0, facesInfo_rgb);
					Rect maxFace_rgb = FaceInfo2Rect(maxFaceInfo_rgb);
					//没在指定矩形框内不检测
					if (!util::isInside(maxFace_rgb, detectFaceArea))
						continue;

					//电子屏幕攻击(也有可能是人脸检测算法在近红外上精度比较低)
					if (facesInfo_ir.empty())
					{
						cv::putText(org_rgb_cameraFrame_0, "spoofing(ir_empty)", cv::Point(280, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
						cout << "non-living(ir_empty)" << endl;
						continue;
					}
					
				}

				//可能黑暗条件
				if (facesInfo_rgb.empty() && !facesInfo_ir.empty())
				{
					continue;
				}

				FaceInfo maxFaceInfo_rgb = drawRectangle(org_rgb_cameraFrame_0, facesInfo_rgb);
				//for (int i = 0; i < facesInfo_rgb.size(); i++)
				//{
				//	FaceInfo faceInfo = facesInfo_rgb[i];
				//	faceInfo.bbox.xmin += 20;
				//	faceInfo.bbox.xmax += 20;
				//	faceInfo.bbox.ymin -= 10;
				//	faceInfo.bbox.ymax -= 10;

				//	/*float _xmin = faceInfo.bbox.xmin + 20;
				//	float _xmax = faceInfo.bbox.xmax + 20;
				//	float _ymin = faceInfo.bbox.ymin - 10;
				//	float _ymax = faceInfo.bbox.ymax - 10;
				//	FaceInfo info;*/
				//	facesInfo_ir.push_back(faceInfo);
				//}
				FaceInfo maxFaceInfo_ir = drawRectangle(org_ir_cameraFrame_0, facesInfo_ir);

				Rect maxFace_rgb = FaceInfo2Rect(maxFaceInfo_rgb);
				Rect maxFace_ir = FaceInfo2Rect(maxFaceInfo_ir);

				//没在指定矩形框内不检测
				if (!util::isInside(maxFace_rgb, detectFaceArea) || !util::isInside(maxFace_ir, detectFaceArea))
					continue;

				FaceSize rgb_faceSize = getFaceSize(maxFaceInfo_rgb);
				//太小的人脸不检测
				if (rgb_faceSize.height >= predict_img_width && rgb_faceSize.width >= predict_img_height) 
				{
					//TODO: 可见光与近红外人脸IOU低于阈值，则判断为非活体


					//稠密光流法
					if (rgb_cameraFrames.size() > 1)
					{
						Rect rgb_0_cropInfo;
						cropFace4Flow(rgb_cameraFrame_0, maxFaceInfo_rgb, rgb_0_cropInfo);
						Mat rgb_prev_crop_img = rgb_cameraFrame_0(rgb_0_cropInfo);

						for (int img_size_i = 0; img_size_i < train_img_heights.size(); img_size_i++)
						{
							int width = train_img_widths[img_size_i];
							int height = train_img_heights[img_size_i];

							if (!is_train)
							{
								if (width != predict_img_width || height != predict_img_height)
									continue;
							}

							resize(rgb_prev_crop_img, rgb_prev_crop_img, Size(width, height));
							Mat rgb_prev_crop_img_gray;
							cvtColor(rgb_prev_crop_img, rgb_prev_crop_img_gray, CV_BGR2GRAY);

							Mat rgb_flow, rgb_motion2color;
							for (int i = 1; i < rgb_cameraFrames.size(); i++)
							{
								Mat rgb_cur_crop_img = rgb_cameraFrames.at(i)(rgb_0_cropInfo);
								resize(rgb_cur_crop_img, rgb_cur_crop_img, Size(width, height));
								Mat rgb_cur_crop_img_gray;
								cvtColor(rgb_cur_crop_img, rgb_cur_crop_img_gray, CV_BGR2GRAY);

								calcOpticalFlowFarneback(rgb_prev_crop_img_gray, rgb_cur_crop_img_gray, rgb_flow, 0.5, 3, 15, 3, 5, 1.2, 0);

								if (!rgb_flow.empty())
								{
									motionToColor(rgb_flow, rgb_motion2color);
									motionToVectorField(rgb_prev_crop_img, rgb_flow);
									imshow("rgb_motion2color" + img_size_i, rgb_motion2color);
									imshow("rgb_prev_crop_img"+ img_size_i, rgb_prev_crop_img);

									std::swap(rgb_prev_crop_img_gray, rgb_cur_crop_img_gray);

									int iResponse = predict(predictor, rgb_motion2color);

									stringstream str_fResponse;
									str_fResponse << iResponse;
									cv::putText(org_rgb_cameraFrame_0, "fResponse:" + str_fResponse.str(), cv::Point(450, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

									if (1 == iResponse)
									{
										cv::putText(org_rgb_cameraFrame_0, "pass(flow)", cv::Point(280, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
									}
									else
									{
										cv::putText(org_rgb_cameraFrame_0, "spoofing(flow)", cv::Point(280, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
										cout << "non-living" << endl;
									}

									if (startSampling)
									{
										if (samplePositiveData)
										{
											util::gatherDataSet(rgb_motion2color, "./data/c" + to_string(width) + "x" + to_string(height) + "/1/");
										}
										else
										{
											util::gatherDataSet(rgb_motion2color, "./data/c" + to_string(width) + "x" + to_string(height) + "/0/");
										}
									}
								}
							}
						}

					}
				}
			}
			/*else
			{
				cv::putText(org_rgb_cameraFrame_0, "non-living", cv::Point(280, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
				cout << "non-living" << endl;
			}*/



			string showCtrlSs = "startSampling:";
			stringstream startSampling_ss, samplePositiveData_ss, is_train_ss;
			startSampling_ss << startSampling;
			showCtrlSs += startSampling_ss.str();
			samplePositiveData_ss << samplePositiveData;
			showCtrlSs += " ,samplePositiveData:" + samplePositiveData_ss.str();
			is_train_ss << is_train;
			showCtrlSs += " ,is_train:" + is_train_ss.str();
			cv::putText(org_rgb_cameraFrame_0, showCtrlSs, cv::Point(20, org_rgb_cameraFrame_0.rows - 25), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
			cv::imshow("rgb_cameraFrame_0", org_rgb_cameraFrame_0);
			cv::imshow("ir_cameraFrame_0", org_ir_cameraFrame_0);


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

			if (char(c) == 't') // is_train
			{
				is_train = !is_train;
			}

		}
	}
	catch (Exception& e)
	{
		std::cout << e.what() << std::endl;
	}


	return 0;
}