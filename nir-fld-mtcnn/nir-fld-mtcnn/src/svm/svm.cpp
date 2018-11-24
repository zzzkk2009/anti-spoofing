#include <svm/svm.h>

using namespace cv;
using namespace std;

void zk_svm::train(string pos_fileName, string neg_fileName, string save_model_name)
{

	int pos_sampleNum = util::countFileLines(pos_fileName);
	int start_index = pos_fileName.find_last_of("_") + 1;
	int end_index = pos_fileName.rfind(".");
	string featureNum_str = pos_fileName.substr(start_index, end_index - start_index);
	int featureNum = stoi(featureNum_str);
	int pos_start_index = pos_fileName.find_last_of("/") + 1;
	int pos_end_index = pos_fileName.rfind("_");
	string pos_labelName = pos_fileName.substr(pos_start_index, pos_end_index - pos_start_index);

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC); // 可以处理非线性分割的问题
	svm->setKernel(cv::ml::SVM::LINEAR); //径向基函数
	svm->setGamma(0.01);
	svm->setC(10.0);

	//Mat trainingDataMat = Mat::zeros(sampleNum, featureNum, CV_32FC1); //Mat结构
	Mat trainingDataMat;
	Mat trainingLabelsMat;

	ifstream ifs(pos_fileName), neg_ifs(neg_fileName);
	string line;

	if (ifs)
	{
		while (getline(ifs, line)) // line中不包括每行的换行符
		{
			Mat feature = Mat::zeros(1, featureNum, CV_32FC1);
			Mat feature_label(1, 1, CV_32S, {1});
			std::vector<string> fea_v = util::split(line, ',');
			std::vector<float> f(fea_v.size());
			for (int i = 0; i < fea_v.size(); i++)
			{
				f[i] = stof(fea_v[i]);
			}
			memcpy(feature.data, f.data(), f.size() * sizeof(float));
			trainingDataMat.push_back(feature);
			trainingLabelsMat.push_back(feature_label);
		}
	}

	if (neg_ifs)
	{
		while (getline(neg_ifs, line)) // line中不包括每行的换行符
		{
			Mat feature = Mat::zeros(1, featureNum, CV_32FC1);
			Mat feature_label(1, 1, CV_32S, { -1 });
			vector<string> fea_v = util::split(line, ',');
			vector<float> f(fea_v.size());
			for (int i = 0; i < fea_v.size(); i++)
			{
				f[i] = stof(fea_v[i]);
			}
			memcpy(feature.data, f.data(), f.size() * sizeof(float));
			trainingDataMat.push_back(feature);
			trainingLabelsMat.push_back(feature_label);
		}
	}

	//算法终止条件
	svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));

	//训练支持向量
	svm->train(trainingDataMat, ml::SampleTypes::ROW_SAMPLE, trainingLabelsMat);

	//保存训练器
	svm->save(save_model_name);
}

Ptr<cv::ml::SVM> zk_svm::load(string modelFile)
{	
	//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>(modelFile);
	svm->load(modelFile);
	return svm;
}

float zk_svm::predict(Ptr<cv::ml::SVM> model, Mat data)
{
	float fResponse = model->predict(data);
	return fResponse;
}

//int main()
//{
//	zk_svm::train("./data/positive_12321.txt", "./data/negative_12321.txt", "flow_12321.xml");
//	return 0;
//}

