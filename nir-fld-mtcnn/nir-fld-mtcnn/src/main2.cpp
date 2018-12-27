#include <detector/detector.h>

using namespace detect;

int main(int argc, char **argv)
{
	Detector detector = Detector();

	VideoCapture rgb_camera(0);
	VideoCapture ir_camera(1);

	while (true)
	{
		try
		{
			vector<Mat> rgb_cameraFrames, ir_cameraFrames;
			util::getFrames(rgb_camera, ir_camera, rgb_cameraFrames, ir_cameraFrames, 2, 0);

			Mat rgb_cameraFrame_0 = rgb_cameraFrames.at(0);
			Mat org_rgb_cameraFrame_0 = rgb_cameraFrame_0.clone();

			int iResponse = detector.detectSpoofing(rgb_cameraFrames, ir_cameraFrames);

			Mat showImg_rgb = detector.getShowImgRGB();
			Mat showImg_ir = detector.getShowImgIR();

			stringstream str_fResponse;
			str_fResponse << iResponse;
			cv::putText(showImg_rgb, "iResponse:" + str_fResponse.str(), cv::Point(500, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

			cv::imshow("showImg_rgb", showImg_rgb);
			cv::imshow("showImg_ir", showImg_ir);

			int c = waitKey(1);
			if (27 == c) // esc
			{
				break;
			}
			if (32 == c) //¿Õ¸ñ
			{
				detector.setStartSampling(!detector.getStartSampling());
			}
			if (char(c) == 'p') // samplePositiveData
			{
				detector.setSamplePositiveData(!detector.getSamplePositiveData());
			}
		}
		catch (Exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}
	
}