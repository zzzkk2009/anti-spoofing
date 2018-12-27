#include <detector/detector.h>

using namespace detect;

int main(int argc, char **argv)
{
	Detector detector = Detector();
	int label = 1;
	string rgb_cate_dir = "E:/srcs/anti-spoofing/data/ai_challenger_short_video/group7-good/" + to_string(label);

	vector<string> rgb_files;
	util::getFiles(rgb_cate_dir, rgb_files);
	detector.setStartSampling(true);
	detector.setSamplePositiveData(label);

	for (int i = 0; i < rgb_files.size(); i++)
	{
		try
		{
			VideoCapture rgb_camera(rgb_files.at(i));

			while (true)
			{
				vector<Mat> rgb_cameraFrames;
				util::getFrames(rgb_camera, rgb_cameraFrames, 2, 0);

				if (rgb_cameraFrames.size() < 2) { //至少两帧
					break;
				}

				Mat rgb_cameraFrame_0 = rgb_cameraFrames.at(0);

				int iResponse = detector.detectSpoofing4(rgb_cameraFrames);

				Mat showImg_rgb = detector.getShowImgRGB();

				stringstream str_fResponse;
				str_fResponse << iResponse;
				cv::putText(showImg_rgb, "iResponse:" + str_fResponse.str(), cv::Point(500, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

				cv::imshow("showImg_rgb", showImg_rgb);

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