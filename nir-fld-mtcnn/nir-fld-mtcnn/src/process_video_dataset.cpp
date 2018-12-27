#include <detector/detector.h>

using namespace detect;

int main(int argc, char **argv)
{
	Detector detector = Detector();
	int label = 1;
	string rgb_cate_dir = "./videos/2018121216/" + to_string(label) + "/rgbs/";
	string nir_cate_dir = "./videos/2018121216/" + to_string(label) + "/nirs/";

	vector<string> rgb_files;
	vector<string> nir_files;
	util::getFiles(rgb_cate_dir, rgb_files);
	util::getFiles(nir_cate_dir, nir_files);
	detector.setStartSampling(true);
	detector.setSamplePositiveData(label);

	for (int i = 0; i < rgb_files.size(); i++)
	{
		try
		{
			VideoCapture rgb_camera(rgb_files.at(i));
			VideoCapture ir_camera(nir_files.at(i));

			while (true)
			{
				vector<Mat> rgb_cameraFrames, ir_cameraFrames;
				util::getFrames(rgb_camera, ir_camera, rgb_cameraFrames, ir_cameraFrames, 2, 1);

				if (ir_cameraFrames.size() < 2) { //至少两帧
					break;
				}

				Mat rgb_cameraFrame_0 = rgb_cameraFrames.at(0);
				Mat org_rgb_cameraFrame_0 = rgb_cameraFrame_0.clone();
				Mat ir_cameraFrame_0 = ir_cameraFrames.at(0);
				Mat org_ir_cameraFrame_0 = ir_cameraFrame_0.clone();

				vector<FaceInfo> facesInfo_ir = detector.detectFace(ir_cameraFrame_0);
				FaceInfo maxFaceInfo_ir = drawRectangle(org_ir_cameraFrame_0, facesInfo_ir);

				vector<Mat> rgb_motion2colors, ir_motion2colors;
				int type = 0;
				detector.getFlowMotion2Colors(rgb_cameraFrames, ir_cameraFrames, type, maxFaceInfo_ir, maxFaceInfo_ir, rgb_motion2colors, ir_motion2colors);
				for (int i = 0; i < ir_motion2colors.size(); i++) {
					Mat ir_motion2color = ir_motion2colors.at(i);
					imshow("ir_motion2color_" + i, ir_motion2color);
					detector.gatherFlowDataSet(ir_motion2color, 1);
				}

				cv::imshow("org_rgb_cameraFrame_0", org_rgb_cameraFrame_0);
				cv::imshow("org_ir_cameraFrame_0", org_ir_cameraFrame_0);

				int c = waitKey(1);
				if (27 == c) // esc
				{
					return 0;
				}
				if (32 == c) //空格
				{
					detector.setStartSampling(!detector.getStartSampling());
				}
				if (char(c) == 'p') // samplePositiveData
				{
					detector.setSamplePositiveData(!detector.getSamplePositiveData());
				}
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
//	Detector detector = Detector();
//	int label = 0;
//	string rgb_cate_dir = "./videos/2018121815/" + to_string(label) + "/rgbs/";
//	string nir_cate_dir = "./videos/2018121815/" + to_string(label) + "/nirs/";
//
//	vector<string> rgb_files;
//	vector<string> nir_files;
//	util::getFiles(rgb_cate_dir, rgb_files);
//	util::getFiles(nir_cate_dir, nir_files);
//	detector.setStartSampling(true);
//	detector.setSamplePositiveData(label);
//
//	for(int i = 0; i < rgb_files.size(); i++)
//	{
//		try
//		{
//			VideoCapture rgb_camera(rgb_files.at(i));
//			VideoCapture ir_camera(nir_files.at(i));
//
//			while (true)
//			{
//				vector<Mat> rgb_cameraFrames, ir_cameraFrames;
//				util::getFrames(rgb_camera, ir_camera, rgb_cameraFrames, ir_cameraFrames, 2, 1);
//
//				if (ir_cameraFrames.size() < 2) { //至少两帧
//					break;
//				}
//
//				Mat rgb_cameraFrame_0 = rgb_cameraFrames.at(0);
//				Mat org_rgb_cameraFrame_0 = rgb_cameraFrame_0.clone();
//
//				int iResponse = detector.detectSpoofing6(rgb_cameraFrames, ir_cameraFrames);
//
//				Mat showImg_rgb = detector.getShowImgRGB();
//				Mat showImg_ir = detector.getShowImgIR();
//
//				stringstream str_fResponse;
//				str_fResponse << iResponse;
//				cv::putText(showImg_rgb, "iResponse:" + str_fResponse.str(), cv::Point(500, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
//
//				cv::imshow("showImg_rgb", showImg_rgb);
//				cv::imshow("showImg_ir", showImg_ir);
//
//				int c = waitKey(1);
//				if (27 == c) // esc
//				{
//					return 0;
//				}
//				if (32 == c) //空格
//				{
//					detector.setStartSampling(!detector.getStartSampling());
//				}
//				if (char(c) == 'p') // samplePositiveData
//				{
//					detector.setSamplePositiveData(!detector.getSamplePositiveData());
//				}
//			}
//			
//		}
//		catch (Exception& e)
//		{
//			std::cout << e.what() << std::endl;
//		}
//	}
//
//}