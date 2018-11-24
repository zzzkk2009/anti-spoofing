#include <utils/util.h>

using namespace std;

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