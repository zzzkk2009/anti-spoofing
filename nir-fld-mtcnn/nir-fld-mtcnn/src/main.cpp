
#include <mtcnn/mtcnn.h>
#include <flow/flow.h>
#include <utils/util.h>
#include <net/mx_shuffle_predict_2.h>

using namespace cv;
using namespace std;
using namespace flow;

int main(int argc, char **argv)
{
	bool startSampling = false;
	bool samplePositiveData = true;
	bool is_train = false;

	//flow:shufflenet
	//Create predictor
	string prefix = "./model/mx-flow-shuffle/";
	string symbol_file = prefix + "checkpoint_c48x64_1w5k/checkpoint_c48x64_1w5k-symbol.json";
	string params_file = prefix + "checkpoint_c48x64_1w5k/checkpoint_c48x64_1w5k-1000.params";
	string synset_file = prefix + "synset.txt";
	PredictorHandle predictor = 0;
	vector<int> train_img_heights = {64}; // {32, 64}
	vector<int> train_img_widths = { 48 }; // { 32, 48 }
	mx_uint predict_img_height = 64, predict_img_width = 48;
	int status = initPredictor(predictor, symbol_file, params_file, predict_img_height, predict_img_width);
	// Load synsets
	vector<string> synsets = loadSynsets(synset_file.c_str());

	float factor = 0.709f;
	float threshold[3] = { 0.7f, 0.6f, 0.6f };
	int minSize = 72; //minSize对应最小人脸尺寸:~(w x h）;72->~50x70, 96->~58x80, 120->~80x120, 240->~160x200
	MTCNN detector("./model/mtcnn");

	VideoCapture rgb_camera(0);
	VideoCapture ir_camera(1);

	cv::TickMeter tm;

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

			tm.reset();
			tm.start();

			double t = (double)cv::getTickCount();

			Mat displayedFrame(rgb_cameraFrame_0.size(), CV_8UC3);

			vector<FaceInfo> facesInfo_rgb = detector.Detect_mtcnn(rgb_cameraFrame_0, minSize, threshold, factor, 3);
			vector<FaceInfo> facesInfo_ir = detector.Detect_mtcnn(ir_cameraFrame_0, minSize, threshold, factor, 3);

			std::cout << "detect time," << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s"
				<< std::endl;

			tm.stop();
			int fps = 1000.0 / tm.getTimeMilli();
			std::stringstream ss;
			ss << fps;
			cv::putText(org_rgb_cameraFrame_0, ss.str() + "FPS",
				cv::Point(20, 45), 4, 0.5, cv::Scalar(0, 0, 125));

			FaceInfo maxFaceInfo_rgb = drawRectangle(org_rgb_cameraFrame_0, facesInfo_rgb);
			FaceInfo maxFaceInfo_ir = drawRectangle(org_ir_cameraFrame_0, facesInfo_ir);

			if (facesInfo_rgb.size() > 0 && facesInfo_ir.size() > 0 && facesInfo_rgb.size() == facesInfo_ir.size())
			{

			}
			else
			{
				//cv::putText(rgb_cameraFrame, "non-living", cv::Point(280, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
				cout << "non-living" << endl;
			}


			if (maxFaceInfo_rgb.bbox.score > 0)
			{
				FaceSize rgb_faceSize = getFaceSize(maxFaceInfo_rgb);
				if (rgb_faceSize.height >= 64 && rgb_faceSize.width >= 48)
				{
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
									//imshow("rgb_prev_crop_img"+ img_size_i, rgb_prev_crop_img);

									std::swap(rgb_prev_crop_img_gray, rgb_cur_crop_img_gray);

									string label_synset_str = "";
									int iResponse = predict(predictor, synsets, rgb_motion2color, label_synset_str, predict_img_height, predict_img_width);

									stringstream str_fResponse;
									str_fResponse << iResponse;
									cv::putText(org_rgb_cameraFrame_0, "fResponse:" + str_fResponse.str(), cv::Point(450, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

									if (1 == iResponse)
									{
										cv::putText(org_rgb_cameraFrame_0, "living(flow)", cv::Point(280, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
									}
									else
									{
										cv::putText(org_rgb_cameraFrame_0, "non-living(flow)", cv::Point(280, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
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