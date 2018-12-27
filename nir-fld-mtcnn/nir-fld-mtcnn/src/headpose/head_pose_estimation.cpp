#include <headpose/head_pose_estimation.h>

using namespace cv;
using namespace std;
using namespace hpe;

//http://www.voidcn.com/article/p-nduikump-brp.html
//本篇主要记录由mtcnn检测得的关键点作人头姿态估计，思路较为简单，mtcnn是一种可以检测输出5个关键点的人脸检测算法，
//分别是左眼，右眼，鼻尖，嘴的左角和嘴的右角。当获得图像中人脸的5个2D关键点，
//再由Opencv中POSIT的姿态估计算法将5个世界坐标系的模板3D关键点通过旋转、平移等变换投射至这5个2D关键点，
//进而估计得变换参数，最后求得2D平面中的人头的姿态参数，
//分别为Yaw:摇头  左正右负、Pitch : 点头 上负下正、Roll : 摆头（歪头）左负 右正
namespace hpe {
	void rot2Euler(const Mat& rotation3_3, HeadPose& headPose) {
		float q0 = sqrt(1+rotation3_3.at<double>(0,0) + rotation3_3.at<double>(1,1) + rotation3_3.at<double>(2,2)) / 2;
		float q1 = (rotation3_3.at<double>(2, 1) - rotation3_3.at<double>(1, 2)) / (4 * q0);
		float q2 = (rotation3_3.at<double>(0, 2) - rotation3_3.at<double>(2, 0)) / (4 * q0);
		float q3 = (rotation3_3.at<double>(1, 0) - rotation3_3.at<double>(0, 1)) / (4 * q0);

		// Slower, but dealing with degenerate cases due to precision
		float t1 = 2.0f * (q0*q2 + q1 * q3);
		if (t1 > 1) t1 = 1.0f;
		if (t1 < -1) t1 = -1.0f;

		float yaw = asin(t1);
		float pitch = atan2(2.0f*(q0*q1 - q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3);
		float roll = atan2(2.0f * (q0*q3 - q1 * q2), q0*q0 + q1 * q1 - q2 * q2 - q3 * q3);
		headPose.yaw = isnan(yaw) ? 0.0f : yaw;
		headPose.pitch = isnan(pitch) ? 0.0f : pitch;
		headPose.roll = isnan(roll) ? 0.0f : roll;
	}
	void headPoseEstimate(Mat& faceImg, const vector<Point2d>& facial5Pts, HeadPose& headPose) {
		// 3D model points
		vector<Point3f> model_points, object_pts;
		model_points.push_back(Point3d(-165.0f, 170.0f, -115.0f)); // left eye
		model_points.push_back(Point3d(165.0f, 170.0f, -115.0f)); // right eye
		model_points.push_back(Point3d(0.0, 0.0, 0.0)); // nose tip
		model_points.push_back(Point3d(-150.0f, -150.0f, -125.0f)); // left mouth corner
		model_points.push_back(Point3d(150.0f, -150.0f, -125.0f)); // right mouth corner

		//object_pts.push_back(cv::Point3d(3.311432, 5.485328, 3.987654));     //#13 left eye
		//object_pts.push_back(cv::Point3d(-3.789930, 5.393625, 4.413414));    //#25 right eye
		//object_pts.push_back(cv::Point3d(0, 1.409845, 6.165652));			 //#55 nose
		//object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
		//object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner

		// Camera internals
		int fx = faceImg.cols;
		int fy = faceImg.rows;
		Point2d center = Point2d(faceImg.cols / 2, faceImg.rows / 2);
		Mat camera_matrix = (Mat_<double>(3,3) << fx, 0, center.x, 0, fy, center.y, 0, 0, 1);
		Mat dist_coeffs = Mat::zeros(4, 1, DataType<double>::type);
		Mat rotation_vector;
		Mat translation_vector;
		solvePnP(model_points, facial5Pts, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

		/*投影一条直线而已
		std::vector<Point3d> nose_end_point3D;
		std::vector<Point2d> nose_end_point2D;
		nose_end_point3D.push_back(cv::Point3d(0,0,1000.0));

			projectPoints(nose_end_point3D, rotation_vector, translation_vector,camera_matrix, dist_coeffs, nose_end_point2D);
		//std::cout << "Rotation Vector " << std::endl << rotation_vector << std::endl;
		//std::cout << "Translation Vector" << std::endl << translation_vector << std::endl;

		cv::Mat img(faceImg);
		cv::line(img ,facial5Pts[2], nose_end_point2D[0], cv::Scalar(255,0,0), 2);
			cv::imshow("vvvvvvvv" ,img );
		cv::waitKey(1);  */

		Mat rotation3_3;
		Rodrigues(rotation_vector, rotation3_3);
		rot2Euler(rotation3_3, headPose);
	}

	Mat headPoseEstimate(Mat& img, FaceInfo& faceInfo, HeadPose& headPose) {
		int xmin = faceInfo.bbox.xmin;
		int ymin = faceInfo.bbox.ymin;
		int xmax = faceInfo.bbox.xmax;
		int ymax = faceInfo.bbox.ymax;

		Mat faceImg = img(Rect(xmin, ymin, xmax - xmin, ymax - ymin));
		vector<Point2d> face5Pts; // 脸部5个点的坐标，原点坐标为(0, 0)
		for (int i = 0; i < 10;) {
			face5Pts.push_back(Point2f(faceInfo.landmark[i], faceInfo.landmark[i + 1]));
			i += 2;
		}
		headPoseEstimate(faceImg, face5Pts, headPose);
		return faceImg;
	}

	vector<Mat> headPoseEstimate(Mat& img, vector<FaceInfo>& faceInfo, vector<HeadPose>& headPoses) {
		vector<Mat> faceImgs;
		for (auto it = faceInfo.begin(); it != faceInfo.end(); it++) {
			HeadPose headPose;
			Mat faceImg = headPoseEstimate(img, (*it), headPose);
			headPoses.push_back(headPose);
			faceImgs.push_back(faceImg);
		}
		return faceImgs;
	}

	void showheadPose(Mat& img, FaceInfo& faceInfo) {
		
		HeadPose headPose;
		Mat faceImg = headPoseEstimate(img, faceInfo, headPose);
		float yaw = headPose.yaw;
		float pitch = headPose.pitch;
		float roll = headPose.roll;
		cout << "yaw:" << yaw << ",pitch:" << pitch << ",roll:" << roll << endl;
		char ch[20];
		sprintf(ch, "yaw:%0.4f", yaw);
		putText(faceImg, ch, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 2, 3);
		sprintf(ch, "pitch:%0.4f", pitch);
		putText(faceImg, ch, Point(20, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 2, 3);
		sprintf(ch, "roll:%0.4f", roll);
		putText(faceImg, ch, Point(20, 120), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 2, 3);
		imshow("faceImg", faceImg);
	}

	void headPoseEstimation(Mat& img, FaceInfo& faceInfo) {
		//Intrisics can be calculated using opencv sample code under opencv/sources/samples/cpp/tutorial_code/calib3d
		//Normally, you can also apprximate fx and fy by image width, cx by half image width, cy by half image height instead
		//double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
		double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };

		int xmin = faceInfo.bbox.xmin;
		int ymin = faceInfo.bbox.ymin;
		int xmax = faceInfo.bbox.xmax;
		int ymax = faceInfo.bbox.ymax;
		Mat faceImg = img(Rect(xmin, ymin, xmax - xmin, ymax - ymin));

		// Camera internals
		double fx = img.cols;
		double fy = img.rows;
		Point2d center = Point2d(img.cols / 2, img.rows / 2);
		double K[9] = { fx, 0, center.x, 0, fy, center.y, 0, 0, 1 };
		/*Mat camera_matrix = (Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
		Mat dist_coeffs = Mat::zeros(4, 1, DataType<double>::type);*/

		//fill in cam intrinsics and distortion coefficients
		cv::Mat cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
		cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);

		//fill in 3D ref points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
		std::vector<cv::Point3d> object_pts;
		object_pts.push_back(cv::Point3d(3.311432, 5.485328, 3.987654));     //#13 left eye
		object_pts.push_back(cv::Point3d(-3.789930, 5.393625, 4.413414));    //#25 right eye
		object_pts.push_back(cv::Point3d(0, 1.409845, 6.165652));			 //#55 nose
		object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
		object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner

		//vector<Point3f> model_points;
		//model_points.push_back(Point3d(-165.0f, 170.0f, -115.0f)); // left eye
		//model_points.push_back(Point3d(165.0f, 170.0f, -115.0f)); // right eye
		//model_points.push_back(Point3d(0.0, 0.0, 0.0)); // nose tip
		//model_points.push_back(Point3d(-150.0f, -150.0f, -125.0f)); // left mouth corner
		//model_points.push_back(Point3d(150.0f, -150.0f, -125.0f)); // right mouth corner

		//object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
		//object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
		//object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
		//object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
		//object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
		//object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
		//object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
		//object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
		//object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
		//object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
		//object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
		//object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
		//object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
		//object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner

		//2D ref points(image coordinates), referenced from detected facial feature
		std::vector<cv::Point2d> image_pts;

		//result
		cv::Mat rotation_vec;                           //3 x 1
		cv::Mat rotation_mat;                           //3 x 3 R
		cv::Mat translation_vec;                        //3 x 1 T
		cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
		cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);

		//reproject 3D points world coordinate axis to verify result pose
		std::vector<cv::Point3d> reprojectsrc;
		reprojectsrc.push_back(cv::Point3d(10.0, 10.0, 10.0));
		reprojectsrc.push_back(cv::Point3d(10.0, 10.0, -10.0));
		reprojectsrc.push_back(cv::Point3d(10.0, -10.0, -10.0));
		reprojectsrc.push_back(cv::Point3d(10.0, -10.0, 10.0));
		reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, 10.0));
		reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, -10.0));
		reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, -10.0));
		reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, 10.0));

		//reprojected 2D points
		std::vector<cv::Point2d> reprojectdst;
		reprojectdst.resize(8);

		//img buf for decomposeProjectionMatrix()
		cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
		cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
		cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);

		//text on screen
		std::ostringstream outtext;

		//fill in 2D ref points, annotations follow https://ibug.doc.ic.ac.uk/resources/300-W/
		image_pts.push_back(cv::Point2d(faceInfo.landmark[0], faceInfo.landmark[1])); //#36 left eye
		image_pts.push_back(cv::Point2d(faceInfo.landmark[2], faceInfo.landmark[3])); //#39 right eye
		image_pts.push_back(cv::Point2d(faceInfo.landmark[4], faceInfo.landmark[5])); //#42 nose
		image_pts.push_back(cv::Point2d(faceInfo.landmark[6], faceInfo.landmark[7])); //#45 mouth left corner
		image_pts.push_back(cv::Point2d(faceInfo.landmark[8], faceInfo.landmark[9])); //#31 mouth right corner

		//image_pts.push_back(cv::Point2d(shape.part(17).x(), shape.part(17).y())); //#17 left brow left corner
		//image_pts.push_back(cv::Point2d(shape.part(21).x(), shape.part(21).y())); //#21 left brow right corner
		//image_pts.push_back(cv::Point2d(shape.part(22).x(), shape.part(22).y())); //#22 right brow left corner
		//image_pts.push_back(cv::Point2d(shape.part(26).x(), shape.part(26).y())); //#26 right brow right corner
		//image_pts.push_back(cv::Point2d(shape.part(36).x(), shape.part(36).y())); //#36 left eye left corner
		//image_pts.push_back(cv::Point2d(shape.part(39).x(), shape.part(39).y())); //#39 left eye right corner
		//image_pts.push_back(cv::Point2d(shape.part(42).x(), shape.part(42).y())); //#42 right eye left corner
		//image_pts.push_back(cv::Point2d(shape.part(45).x(), shape.part(45).y())); //#45 right eye right corner
		//image_pts.push_back(cv::Point2d(shape.part(31).x(), shape.part(31).y())); //#31 nose left corner
		//image_pts.push_back(cv::Point2d(shape.part(35).x(), shape.part(35).y())); //#35 nose right corner
		//image_pts.push_back(cv::Point2d(shape.part(48).x(), shape.part(48).y())); //#48 mouth left corner
		//image_pts.push_back(cv::Point2d(shape.part(54).x(), shape.part(54).y())); //#54 mouth right corner
		//image_pts.push_back(cv::Point2d(shape.part(57).x(), shape.part(57).y())); //#57 mouth central bottom corner
		//image_pts.push_back(cv::Point2d(shape.part(8).x(), shape.part(8).y()));   //#8 chin corner

		//calc pose
		cv::solvePnP(object_pts, image_pts, cam_matrix, Mat(), rotation_vec, translation_vec);

		//reproject
		cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, Mat(), reprojectdst);

		//draw axis
		cv::line(img, reprojectdst[0], reprojectdst[1], cv::Scalar(0, 0, 255));
		cv::line(img, reprojectdst[1], reprojectdst[2], cv::Scalar(0, 0, 255));
		cv::line(img, reprojectdst[2], reprojectdst[3], cv::Scalar(0, 0, 255));
		cv::line(img, reprojectdst[3], reprojectdst[0], cv::Scalar(0, 0, 255));
		cv::line(img, reprojectdst[4], reprojectdst[5], cv::Scalar(0, 0, 255));
		cv::line(img, reprojectdst[5], reprojectdst[6], cv::Scalar(0, 0, 255));
		cv::line(img, reprojectdst[6], reprojectdst[7], cv::Scalar(0, 0, 255));
		cv::line(img, reprojectdst[7], reprojectdst[4], cv::Scalar(0, 0, 255));
		cv::line(img, reprojectdst[0], reprojectdst[4], cv::Scalar(0, 0, 255));
		cv::line(img, reprojectdst[1], reprojectdst[5], cv::Scalar(0, 0, 255));
		cv::line(img, reprojectdst[2], reprojectdst[6], cv::Scalar(0, 0, 255));
		cv::line(img, reprojectdst[3], reprojectdst[7], cv::Scalar(0, 0, 255));

		//calc euler angle
		cv::Rodrigues(rotation_vec, rotation_mat);
		cv::hconcat(rotation_mat, translation_vec, pose_mat);
		cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);

		//show angle result
		outtext << "X: " << std::setprecision(3) << euler_angle.at<double>(0);
		cv::putText(img, outtext.str(), cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
		outtext.str("");
		outtext << "Y: " << std::setprecision(3) << euler_angle.at<double>(1);
		cv::putText(img, outtext.str(), cv::Point(50, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
		outtext.str("");
		outtext << "Z: " << std::setprecision(3) << euler_angle.at<double>(2);
		cv::putText(img, outtext.str(), cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
		outtext.str("");

		image_pts.clear();
		//cv::imshow("img", img);
	}
}

