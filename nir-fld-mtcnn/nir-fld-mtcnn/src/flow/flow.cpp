
#include <flow/flow.h>

using namespace cv;
using namespace std;
using namespace flow;

#define UNKNOWN_FLOW_THRESH 1e9

// Color encoding of flow vectors from:
// http://members.shaw.ca/quadibloc/other/colint.htm
// This code is modified from:
// http://vision.middlebury.edu/flow/data/
void flow::makecolorwheel(vector<Scalar> &colorwheel)
{
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int i;

	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

bool flow::isErrorFlow(Mat& flow) {
	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			if (isnan(fx) || isnan(fy) || 0 == fx || 0 == fy ||
				(fabs(fx) > UNKNOWN_FLOW_THRESH) || (fabs(fy) > UNKNOWN_FLOW_THRESH)) {
				return true;
			}
		}
	}
	return false;
}

void flow::motionToColor(Mat& flow, Mat &color)
{
	if (color.empty())
		color.create(flow.rows, flow.cols, CV_8UC3);

	static vector<Scalar> colorwheel; //Scalar r,g,b
	if (colorwheel.empty())
		flow::makecolorwheel(colorwheel);

	// determine motion range:
	float maxrad = -1;

	// Find max flow to normalize fx and fy
	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			if (isnan(fx) || isnan(fy)) {
				continue;
			}
			if ((fabs(fx) > UNKNOWN_FLOW_THRESH) || (fabs(fy) > UNKNOWN_FLOW_THRESH))
				continue;
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = maxrad > rad ? maxrad : rad;
		}
	}
	
	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);

			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;

			if (isnan(fx) || isnan(fy)) {
				data[0] = data[1] = data[2] = 0;
				continue;
			}

			if ((fabs(fx) > UNKNOWN_FLOW_THRESH) || (fabs(fy) > UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI; // angle: [-1, 1]
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;
			//f = 0; // uncomment to see original color wheel
			//cout << "1111" << endl;
			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius
				else
					col *= .75; // out of range
				data[2 - b] = (int)(255.0 * col);
			}
		}
	}
}

void flow::drawArrow(cv::Mat& img, cv::Point& pStart, cv::Point& pEnd, int len, int alpha, cv::Scalar& color, int thickness, int lineType)
{
	const double PI = 3.1415926;
	Point arrow;
	//计算 θ 角（最简单的一种情况在下面图示中已经展示，关键在于 atan2 函数，详情见下面）   
	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));

	line(img, pStart, pEnd, color, thickness, lineType);

	//计算箭角边的另一端的端点位置（上面的还是下面的要看箭头的指向，也就是pStart和pEnd的位置） 
	arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);

	arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);

	line(img, pEnd, arrow, color, thickness, lineType);

	arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);

	arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);

	line(img, pEnd, arrow, color, thickness, lineType);
}

void flow::motionToVectorField(Mat& img, Mat& flow)
{
	Scalar lineColor(0, 255, 0);
	/*int end_i = 0;
	int end_j = 0;*/
	for (int i = 0; i < flow.rows; i++)
	{
		if (i % 5 != 0)
			continue;

		for (int j = 0; j < flow.cols; j++)
		{
			if (j % 5 != 0)
				continue;

			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];

			int arrowLen = 8;

			int end_x = i;
			int end_y = j;

			if (fx != 0)
			{
				end_x = arrowLen / sqrt(1 + pow(fy / fx, 2)) + i;
			}
			if (fy != 0)
			{
				end_y = arrowLen / sqrt(1 + pow(fx / fy, 2)) + j;
			}

			/*end_x = i + fx;
			end_y = j + fy;*/

			/*int end_x = i, end_y = j;
			if (fx > 0)
			{
				end_x += 10;
			}
			else if (fx < 0)
			{
				end_x -= 10;
			}

			if (fy > 0)
			{
				end_y += 10;
			}
			else if (fy < 0)
			{
				end_y -= 10;
			}*/

			if (fx != 0 || fy != 0)
			{
				if (i > 400 && j > 600)
				{
					cout << i;
				}
				Point pStart(i, j);
				Point pEnd(end_x, end_y);
				float len = sqrt(pow(pEnd.x - pStart.x, 2) + pow(pEnd.y - pStart.y, 2));
				int sub_len = len * 0.3 > 10 ?  3 : int(len * 0.3);
				drawArrow(img, pStart, pEnd, sub_len, 45, lineColor);
			}
			//end_j = j;
		}
		//end_i = i;
	}
	/*cout << end_i;
	cout << end_j;*/
}

vector<int> flow::calcFlowAngleHist(Mat& flow, FLOW_HIST_TYPE flowHistType)
{
	int size = 360 / flowHistType;
	vector<int> hist(size);

	try
	{
		for (int i = 0; i < flow.rows; ++i)
		{
			for (int j = 0; j < flow.cols; ++j)
			{
				Vec2f flow_at_point = flow.at<Vec2f>(i, j);
				float fx = flow_at_point[0];
				float fy = flow_at_point[1];
				float angle = atan2(-fy, -fx) * 180 / CV_PI;

				if (angle < 0)
				{
					angle += 360;
				}

				int index = angle / flowHistType;

				int sum = hist.at(index);
				sum += 1;
				if (index < size)
				{
					hist[index] = sum;
				}

			}
		}
	}catch(Exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	
	return hist;
}

vector<float> flow::extractFlowAnglFeature(Mat& flow, SAMPLE_MARGIN sampleMargin)
{
	int size = (flow.rows / sampleMargin + 1) * (flow.cols / sampleMargin + 1);
	vector<float> feature(size);

	for (int i = 0; i < flow.rows; ++i)
	{
		if (i % sampleMargin != 0)
		{
			continue;
		}
		for (int j = 0; j < flow.cols; ++j)
		{
			if (j % sampleMargin != 0)
			{
				continue;
			}
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			float angle = atan2(-fy, -fx) * 180 / CV_PI;

			feature.push_back(angle);

		}
	}

	return feature;
}

void flow::computeLBPFeature(Mat& motion2color, Mat& featureMat)
{
	Mat motion2color_gray;
	cvtColor(motion2color, motion2color_gray, CV_BGR2GRAY);
	UniformRotInvLBPFeature(motion2color_gray, Size(4, 4), featureMat);
	featureMat.convertTo(featureMat, CV_32F);
}

void flow::computeLBPFeature(Mat& motion2color, vector<float>& feature)
{
	Mat featureMat;
	computeLBPFeature(motion2color, featureMat);
	feature.assign((float*)featureMat.datastart, (float*)featureMat.dataend);
}
