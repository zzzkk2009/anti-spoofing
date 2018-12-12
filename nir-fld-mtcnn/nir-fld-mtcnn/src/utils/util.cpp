#include <utils/util.h>

using namespace std;
using namespace util;

double util::computeVariance(vector<double> v)
{
	double sum = accumulate(std::begin(v), std::end(v), 0.0);
	double mean = sum / v.size(); //均值

	double accum = 0.0;
	std::for_each(std::begin(v), std::end(v), [&](const double d) {
		accum += (d - mean)*(d - mean);
	});

	return accum / (v.size() - 1); //方差
}

double util::computeStdDev(vector<double> v)
{
	return sqrt(util::computeVariance(v));
}

void util::writeFile(string filename, vector<float> v)
{
	ofstream ofs;
	ofs.open(filename, ios::app); // ios::out ios::app
	ofs << endl;

	// Print out the histogram values
	double sum = 0;
	//cout << "v = [";
	for (int i = 0; i < v.size(); i++) {
		//cout << v[i] << ", ";
		sum += v[i];

		if (i > 0) ofs << ", ";
		ofs << v[i];
	}
	/*cout << "]; " << endl;
	cout << "v sum=" << sum << endl;*/
		
	ofs << endl;
	ofs << "sum=" << sum;
	ofs << endl;
	ofs.close();
}

ofstream util::getOfstream(string dirName, string fileName)
{
	ofstream ofs;
	string dirPath = "./";

	if (access(dirName.c_str(), 0) == -1) // 目录不存在
	{
		int flag = mkdir(dirName.c_str()); // flag: 0,成功   -1,失败
	}
	dirPath += dirName;

	fileName = dirPath + "/" + fileName;
	
	ofs.open(fileName, ios::app); // ios::out ios::app
	return ofs;
}

void util::saveTrainingData(ofstream* ofs, vector<float> v)
{
	for (int i = 0; i < v.size(); i++) {
		if (i > 0) *ofs << ", ";
		*ofs << v[i];
	}
	*ofs << endl;
	/**ofs << endl;*/
}

int util::countFileLines(string fileName)
{
	ifstream ifs(fileName);
	string line;
	int n = 0;
	if (ifs)
	{
		while (getline(ifs, line)) // line中不包括每行的换行符
		{
			n++;
		}
	}
	return n;
}

void util::preproccessTrainingData(string fileName)
{
	ifstream ifs(fileName);
	string line;

	if (ifs)
	{
		while (getline(ifs, line)) // line中不包括每行的换行符
		{

		}
	}
	else 
	{

	}
}

//static void _split(const std::string &s, char delim,
//	std::vector<std::string> &elems) {
//	std::stringstream ss(s);
//	std::string item;
//
//	while (std::getline(ss, item, delim)) {
//		elems.push_back(item);
//	}
//}

std::vector<std::string> util::split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	//_split(s, delim, elems);

	std::stringstream ss(s);
	std::string item;

	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}

	return elems;
}

void util::cropArea4Flow(Mat& img, Mat& cropedArea)
{
	int clip_x = img.cols * 0.25;
	int clip_y = img.rows * 0.25;
	int clip_w = img.cols * 0.65;
	int clip_h = img.rows * 0.65;

	Rect clipArea = Rect(clip_x, clip_y, clip_w, clip_h);
	cropedArea = img(clipArea);
	resize(cropedArea, cropedArea, Size(150, 150));
}

void util::cropArea4Flow(Mat& img, Rect& cropRect, Mat& cropedArea)
{
	cropedArea = img(cropRect);
}

void util::gatherDataSet(Mat& img, string mainDir)
{
	string nextDirName = util::cf_NextdirName(mainDir);
	string nextDirPath = mainDir + nextDirName;
	string nextFileName = util::cf_NextFileName(nextDirPath);
	string filename = nextDirPath + "/" + nextFileName;
	imwrite(filename, img);
}


void util::cf_findFileFromDir(string mainDir, vector<string> &files)
{
	char* cur_work_dir = getcwd(NULL, NULL);
	files.clear();
	const char *dir = mainDir.c_str();
	_chdir(dir);
	long hFile;
	_finddata_t fileinfo;

	if ((hFile = _findfirst("*.*", &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))//找到文件夹
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					char subdir[_MAX_PATH];
					strcpy_s(subdir, dir);
					//strcat_s(subdir, "\\");
					strcat_s(subdir, "/");
					strcat_s(subdir, fileinfo.name);
					string temDir = subdir;
					vector<string> temFiles;
					cf_findFileFromDir(temDir, temFiles);
					for (vector<string>::iterator it = temFiles.begin(); it < temFiles.end(); it++)
					{
						files.push_back(*it);
					}
				}
			}
			else//直接找到文件
			{
				char filename[_MAX_PATH];
				strcpy_s(filename, dir);
				//strcat_s(filename, "\\");
				strcat_s(filename, "/");
				strcat_s(filename, fileinfo.name);
				string temfilename = filename;
				files.push_back(temfilename);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	_chdir(cur_work_dir);
}

void util::cf_findFileFromDir2(string mainDir, vector<string> &files)
{
	char* cur_work_dir = getcwd(NULL, NULL);
	files.clear();
	const char *dir = mainDir.c_str();
	_chdir(dir);
	long hFile;
	_finddata_t fileinfo;

	if ((hFile = _findfirst("*.*", &fileinfo)) != -1)
	{
		do
		{
			if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
			{
				files.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	_chdir(cur_work_dir);
}

void util::cf_findFileFromDir(string mainDir, vector<int> &files)
{
	vector<string> s_files;
	cf_findFileFromDir(mainDir, s_files);
	if (s_files.size() > 0)
	{
		for (int i = 0; i < s_files.size(); i++)
		{
			files.push_back(stoi(s_files[i]));
		}
	}
}



void util::cf_findSubDirFromDir(string mainDir, vector<int> &dirs)
{
	char* cur_work_dir = getcwd(NULL, NULL);

	if (access(mainDir.c_str(), 0) == -1) // 目录不存在
	{
		int flag = mkdir(mainDir.c_str()); // flag: 0,成功   -1,失败
	}

	dirs.clear();
	const char *dir = mainDir.c_str();
	_chdir(dir);
	long hFile;
	_finddata_t fileinfo;

	if ((hFile = _findfirst("*.*", &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))//找到文件夹
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					dirs.push_back(stoi(fileinfo.name));
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}

	_chdir(cur_work_dir);
}

template <typename T, typename S>
T util::fill_cast(const S& v, const int width, const char c)
{
	T result;
	std::stringstream inter;
	inter << std::setw(width) << std::setfill(c) << v;
	inter >> result;
	return result;
}

string util::cf_NextdirName(string mainDir)
{
	vector <int> dirs;
	cf_findSubDirFromDir(mainDir, dirs);
	int endDir = 1;
	int dirWidth = 5;
	if (dirs.size())
	{
		sort(dirs.begin(), dirs.end());
		endDir = dirs.at(dirs.size() - 1);
		string subMainDir = fill_cast<std::string>(endDir, dirWidth, '0');

		string subMainPath = mainDir + subMainDir;
		//if (access(subMainPath.c_str(), 0) == -1) // 目录不存在
		//{
		//	int flag = mkdir(subMainPath.c_str()); // flag: 0,成功   -1,失败
		//}

		vector<string> subFiles;
		cf_findFileFromDir2(subMainPath, subFiles);
		if (subFiles.size() >= 1000)
		{
			endDir += 1;
		}
	}

	
	string nextDirName = fill_cast<std::string>(endDir, dirWidth, '0');

	string nextDirpath = mainDir + nextDirName;
	if (access(nextDirpath.c_str(), 0) == -1) // 目录不存在
	{
		int flag = mkdir(nextDirpath.c_str()); // flag: 0,成功   -1,失败
	}

	return nextDirName;
}

string util::cf_NextFileName(string dirName)
{
	vector<string> files;
	cf_findFileFromDir2(dirName, files);

	string nextFileName = "1.png";
	if (files.size() > 0)
	{
		sort(files.begin(), files.end());
		string endFile = files.at(files.size() - 1);
		std::vector<string> f_v = split(endFile, '.');
		int iFilename = stoi(f_v[0]);
		iFilename += 1;
		nextFileName = to_string(iFilename) + ".png";
	}

	nextFileName = fill_cast<std::string>(nextFileName, 9, '0');
	return nextFileName;
}

// 带HUB的近红外双摄像头YUV，目前最大支持640x480，帧率最大只有15 FPS
void util::getFrames(VideoCapture& rgb_camera, VideoCapture& ir_camera, vector<Mat>& rgb_cameraFrames, vector<Mat>& ir_cameraFrames, int frame_num, int margin_frame)
{
	if (frame_num < 1)
		frame_num = 1;
	if (margin_frame < 0)
		margin_frame = 0;
	
	rgb_cameraFrames.clear();
	ir_cameraFrames.clear();

	int n = -1;
	while (true)
	{
		n++;

		Mat rgb_cameraFrame, ir_cameraFrame;
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

		if (n % (margin_frame + 1) != 0)
			continue;

		if (rgb_cameraFrames.size() < frame_num)
		{
			rgb_cameraFrames.push_back(rgb_cameraFrame);
			ir_cameraFrames.push_back(ir_cameraFrame);
		}
		else
			break;
	}
}

void util::getDetectFaceArea(Mat& img, Rect& rect)
{
	int height = img.rows;
	int width = img.cols;

	int xmin = width * 0.2;
	int ymin = height * 0.1;
	int xmax = width * 0.8;
	int ymax = height;
	rect = Rect(xmin, ymin, xmax-xmin, ymax-ymin);
}


//判断rect1是否在rect2里面
bool util::isInside(Rect rect1, Rect rect2)
{
	return (rect1 == (rect1&rect2)); // &：交集； |：并集
}

float util::rectIOU(const cv::Rect& rectA, const cv::Rect& rectB) {
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

	/*intersectRect.x = max(rectA.x, rectB.x);
	intersectRect.y = max(rectA.y, rectB.y);
	intersectRect.width = min(rectA.x + rectA.width, rectB.x + rectB.width) - intersectRect.x;
	intersectRect.height = min(rectA.y + rectA.height, rectB.y + rectB.height) - intersectRect.y;*/
	return intersectionPercent;
}


//void util::saveTrainingData(string dirName, vector<float> v, bool positive)
//{
//	ofstream ofs;
//	string dirPath = "./";
//
//	if (access(dirName.c_str(), 0) == -1) // 目录不存在
//	{
//		int flag = mkdir(dirName.c_str()); // flag: 0,成功   -1,失败
//	}
//	dirPath += dirName;
//
//	if (positive)
//	{
//		if (access("positive", 0) == -1) // 目录不存在
//		{
//			int flag = mkdir("positive"); 
//		}
//		dirPath += "/positive/";
//	}
//	else
//	{
//		if (access("negative", 0) == -1) // 目录不存在
//		{
//			int flag = mkdir("negative");
//		}
//		dirPath += "/negative/";
//	}
//
//	_finddata_t file;
//	long lf;
//	vector<string> files;
//	
//	if ((lf = _findfirst(dirPath.c_str(), &file)) == -1) // 目录下没有文件
//	{
//		string filename = dirPath + "0.txt";
//		ofs.open(filename, ios::app); // ios::out ios::app
//	}
//	else
//	{
//		while (_findnext(lf, &file) == 0)
//		{
//			if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
//				continue;
//			files.push_back(file.name);
//		}
//		sort(files.begin(), files.end());
//
//		string endFile = files.at(files.size() - 1);
//		string endFileName = dirPath + endFile;
//		string pureEndFileName = endFileName.substr(0, endFileName.rfind("."));
//
//		
//
//		FILE *pFile = fopen(endFileName.c_str(), "rb");
//		fseek(pFile, 0, SEEK_END); // 先用fseek将文件指针移动到文件末尾
//		int fileSize = ftell(pFile); // 再用ftell获取文件内指针当前的位置，即为文件大小
//
//		if(fileSize > 5000)
//		{
//			fclose(pFile);
//			string nextFileName = to_string((stoi(pureEndFileName) + 1)) + ".txt";
//			ofs.open(nextFileName, ios::app);
//		}
//		else 
//		{
//			ofs.open(endFileName, ios::app);
//		}
//		
//	}
//
//	for (int i = 0; i < v.size(); i++) {
//		if (i > 0) ofs << ", ";
//		ofs << v[i];
//	}
//	ofs << endl;
//
//	
//
//	_findclose(lf);
//	ofs.close();
//}