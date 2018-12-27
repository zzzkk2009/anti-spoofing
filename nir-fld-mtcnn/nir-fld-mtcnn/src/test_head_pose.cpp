
#include <headpose/head_pose_estimation.h>
#include <detector/detector.h>

using namespace detect;
using namespace hpe;


int main()
{
	Detector detector = Detector();
	Mat img = imread("./image/1.jpg");
	vector<FaceInfo> faceInfo = detector.detectFace(img);
	//showheadPose(img, faceInfo[0]);
	headPoseEstimation(img, faceInfo[0]);
	waitKey(0);
	return 0;
}