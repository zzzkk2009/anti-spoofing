
#include <mtcnn/mtcnn.h>
//#include <mtcnn-light/mtcnn.h>

float rectIOU(const cv::Rect& rectA, const cv::Rect& rectB, cv::Rect& intersectRect) {
	if (rectA.x > rectB.x + rectB.width) { return 0.; }
	if (rectA.y > rectB.y + rectB.height) { return 0.; }
	if ((rectA.x + rectA.width) < rectB.x) { return 0.; }
	if ((rectA.y + rectA.height) < rectB.y) { return 0.; }
	float colInt = min(rectA.x + rectA.width, rectB.x + rectB.width) - max(rectA.x, rectB.x);
	float rowInt = min(rectA.y + rectA.height, rectB.y + rectB.height) - max(rectA.y, rectB.y);
	float intersection = colInt * rowInt;
	float areaA = rectA.width * rectA.height;
	float areaB = rectB.width * rectB.height;
	float intersectionPercent = intersection / (areaA + areaB - intersection);

	intersectRect.x = max(rectA.x, rectB.x);
	intersectRect.y = max(rectA.y, rectB.y);
	intersectRect.width = min(rectA.x + rectA.width, rectB.x + rectB.width) - intersectRect.x;
	intersectRect.height = min(rectA.y + rectA.height, rectB.y + rectB.height) - intersectRect.y;
	return intersectionPercent;
}


int main(int argc, char **argv)
{

	float factor = 0.709f;
	float threshold[3] = { 0.7f, 0.6f, 0.6f };
	int minSize = 24; // 12 
	MTCNN detector("./src/mtcnn/model");
	//MTCNN detector("E:/srcs/anti-spoofing/nir-fld-mtcnn/nir-fld-mtcnn/src/mtcnn/model");

	VideoCapture rgb_camera(0);
	VideoCapture ir_camera(1);
	Mat rgb_cameraFrame, ir_cameraFrame;
	std::vector<FaceInfo> rgb_faces, ir_faces;

	cv::TickMeter tm;

	while (true)
	{
		rgb_camera >> rgb_cameraFrame;
		ir_camera >> ir_cameraFrame;

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

		Rect maxFaceInfo_rgb_rect;
		if (maxFaceInfo_rgb.bbox.score > 0)
		{
			int x = (int)maxFaceInfo_rgb.bbox.xmin;
			int y = (int)maxFaceInfo_rgb.bbox.ymin;
			int w = (int)(maxFaceInfo_rgb.bbox.xmax - maxFaceInfo_rgb.bbox.xmin + 1);
			int h = (int)(maxFaceInfo_rgb.bbox.ymax - maxFaceInfo_rgb.bbox.ymin + 1);
			maxFaceInfo_rgb_rect = cv::Rect(x, y, w, h);
			cv::rectangle(rgb_cameraFrame, maxFaceInfo_rgb_rect, cv::Scalar(255, 0, 0), 1);
		}

		FaceInfo maxFaceInfo_ir = FaceInfo{};
		for (int i = 0; i < facesInfo_ir.size(); i++) {
			FaceInfo faceInfo_ir = facesInfo_ir[i];
			int x = (int)faceInfo_ir.bbox.xmin;
			int y = (int)faceInfo_ir.bbox.ymin;
			int w = (int)(faceInfo_ir.bbox.xmax - faceInfo_ir.bbox.xmin + 1);
			int h = (int)(faceInfo_ir.bbox.ymax - faceInfo_ir.bbox.ymin + 1);

			int area = w * h;

			if (0 == i)
			{
				maxFaceInfo_ir = faceInfo_ir;
			}
			else
			{
				int max_w = (int)(maxFaceInfo_ir.bbox.xmax - maxFaceInfo_ir.bbox.xmin + 1);
				int max_h = (int)(maxFaceInfo_ir.bbox.ymax - maxFaceInfo_ir.bbox.ymin + 1);
				int max_area = max_w * max_h;
				if (area > max_area)
				{
					maxFaceInfo_ir = faceInfo_ir;
				}
			}
		}

		Rect maxFaceInfo_ir_rect;
		if (maxFaceInfo_ir.bbox.score > 0)
		{
			int max_ir_x = (int)maxFaceInfo_ir.bbox.xmin;
			int max_ir_y = (int)maxFaceInfo_ir.bbox.ymin;
			int max_ir_w = (int)(maxFaceInfo_ir.bbox.xmax - maxFaceInfo_ir.bbox.xmin + 1);
			int max_ir_h = (int)(maxFaceInfo_ir.bbox.ymax - maxFaceInfo_ir.bbox.ymin + 1);
			maxFaceInfo_ir_rect = cv::Rect(max_ir_x, max_ir_y, max_ir_w, max_ir_h);
			cv::rectangle(ir_cameraFrame, maxFaceInfo_ir_rect, cv::Scalar(255, 0, 0), 1);
		}

		float intersectionPercent;
		Rect intersectRect;
		intersectionPercent = rectIOU(maxFaceInfo_rgb_rect, maxFaceInfo_ir_rect, intersectRect);
		std::stringstream ss_iou;
		ss_iou << intersectionPercent;
		putText(rgb_cameraFrame, "IOU: " + ss_iou.str(), cv::Point(100, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
		cout << "rect IOU: " << intersectionPercent << endl;

		if (intersectionPercent < 0.3)
		{
			putText(rgb_cameraFrame, "non-living", cv::Point(280, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
			cout << "non-living" << endl;
		}
		else
		{
			putText(rgb_cameraFrame, "living", cv::Point(280, 45), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
			cout << "living" << endl;
		}

		cv::imshow("mtcnn_rgb_cameraFrame", rgb_cameraFrame);
		cv::imshow("mtcnn_ir_cameraFrame", ir_cameraFrame);


		int c = waitKey(1);
		if (27 == c) // esc
		{
			break;
		}
	}

	return 0;
}