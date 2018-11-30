//#include <iostream>
//#include <fstream>
//#include <opencv2/opencv.hpp>
//#include <algorithm>
//#include "c_predict_api.h"
//
#include <net/mx_shuffle_predict_2.h>

using namespace std;
using namespace cv;

// Constants
const mx_uint kIMSIZE = 48;
const cv::Scalar IMAGENET_MEAN = cv::Scalar(103.939, 116.779, 123.68); // Imagenet 1k (from Caffe, maybe different in MXNet!)
//const cv::Scalar IMAGENET_MEAN=cv::Scalar(117, 117, 117); // Imagenet 15k

//----------------------------------------------------------------

// Method to read a raw file and store it into a string
string readAllBytes(const char *filename)
{
	ifstream in(filename, ios::in | ios::binary);

	if (!in.is_open())
	{
		cerr << "Error loading binary file: " << filename << endl;
		exit(-1);
	}

	string contents;
	in.seekg(0, std::ios::end);
	contents.resize(in.tellg());
	in.seekg(0, std::ios::beg);
	in.read(&contents[0], contents.size());
	in.close();

	return(contents);
}

//----------------------------------------------------------------

int initPredictor(PredictorHandle &predictor, string symbol_file, string params_file, mx_uint img_height, mx_uint img_width)
{
	// Init predictor parameters
	/*string prefix = "./src/net/model/mx-flow-shuffle/checkpoint";
	string symbol_file = prefix + "-symbol.json";
	string params_file = prefix + "-0060.params";*/
	int dev_type = 1; // 1:CPU, 2:GPU
	int dev_id = 0;
	mx_uint num_input = 1;
	const char* input_keys[] = { "data" };
	const mx_uint input_shape_indptr[] = { 0,4 };
	//const mx_uint input_shape_data[] = { 1,3,kIMSIZE,kIMSIZE }; // batch_size, channel, height, width
	const mx_uint input_shape_data[] = { 1,3,img_height,img_width }; // batch_size, channel, height, width

	// Read symbols file
	string symbols = readAllBytes(symbol_file.c_str());

	// Read params file
	string strparams = readAllBytes(params_file.c_str());

	// Create predictor
	int status = MXPredCreate(symbols.c_str(),
		strparams.c_str(),
		strparams.size(),
		dev_type,
		dev_id,
		num_input,
		input_keys,
		input_shape_indptr,
		input_shape_data,
		&predictor);

	// Check status
	return status;
}

//----------------------------------------------------------------

// Convert the input image to feed the network

Mat preprocess(const cv::Mat& img, int num_channels, cv::Size input_geometry)
{
	Mat sample;

	// Convet color space
	if (img.channels() == 3 && num_channels == 1)
		cvtColor(img, sample, CV_BGR2GRAY);
	else if (img.channels() == 4 && num_channels == 1)
		cvtColor(img, sample, CV_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels == 3)
		cvtColor(img, sample, CV_BGRA2BGR);
	else if (img.channels() == 1 && num_channels == 3)
		cvtColor(img, sample, CV_GRAY2BGR);
	else sample = img;

	// Resize image
	Mat sample_resized;
	resize(sample, sample_resized, input_geometry);

	Mat sample_float;
	if (num_channels == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	// Compute the global mean pixel value and create a mean image filled with these values
	cv::Scalar channel_mean = IMAGENET_MEAN;
	Mat mean = cv::Mat(input_geometry, CV_32FC3, channel_mean);

	// Subtract mean
	Mat sample_normalized;
	cv::subtract(sample_float, mean, sample_normalized);

	// Convert image from BGR,BGR,BGR,... into RRR...+GGG...+BBB...
	vector<Mat> spl;
	split(sample_normalized, spl);

	Mat output(spl[2]);
	output.push_back(spl[1]);
	output.push_back(spl[0]);

	// Return normalized image
	return output;
}

//----------------------------------------------------------------

vector<string> loadSynsets(const char *filename)
{
	ifstream fi(filename);
	if (!fi.is_open())
	{
		cerr << "Error opening file " << filename << endl;
		exit(-1);
	}
	vector<string> output;

	string synset, lemma;
	while (fi >> synset)
	{
		getline(fi, lemma);
		output.push_back(lemma);
	}
	return output;
}

int predict(PredictorHandle predictor, vector<string>& synsets, Mat& img, string label_synset_str, mx_uint img_height, mx_uint img_width)
{
	const int num_channels = 3; // could be read from input_layer->channels;
	//cv::Size input_geometry = cv::Size(kIMSIZE, kIMSIZE);
	cv::Size input_geometry = cv::Size(img_width, img_height);

	// Read and normalize image
	Mat netImage = preprocess(img, num_channels, input_geometry);

	//-- Set Input Image
	//MXPredSetInput(predictor, "data", (float *)netImage.data, kIMSIZE*kIMSIZE*num_channels);
	MXPredSetInput(predictor, "data", (float *)netImage.data, img_width*img_height*num_channels);

	// Run forward
	MXPredForward(predictor);

	// Get output array from the class layer
	const int outputLayerSize = 2;
	const int outputLayerIndex = 0;
	float *output = new float[outputLayerSize];
	MXPredGetOutput(predictor, outputLayerIndex, &output[0], outputLayerSize);

	int indmax = distance(output, max_element(output, output + outputLayerSize));

	if (synsets.size() > 0)
	{
		label_synset_str = synsets[indmax];
	}
	//cout << indmax << synsets[indmax] << endl;

	// Free memory
	delete[] output;

	return indmax;
}


//----------------------------------------------------------------

//int main(int argc, char *argv[])
//{
//	if (argc != 2)
//	{
//		cerr << "Syntax: " << argv[0] << " <imagefile>" << endl;
//		exit(-1);
//	}
//
//	const int num_channels = 3; // could be read from input_layer->channels;
//	cv::Size input_geometry = cv::Size(kIMSIZE, kIMSIZE);
//
//	// Read and normalize image
//	Mat input = imread(argv[1]);
//	Mat netImage = preprocess(input, num_channels, input_geometry);
//
//	// Create predictor
//	PredictorHandle predictor = 0;
//	int status = initPredictor(predictor);
//
//	if (status == 0)
//	{
//		//-- Set Input Image
//		MXPredSetInput(predictor, "data", (float *)netImage.data, kIMSIZE*kIMSIZE*num_channels);
//
//		// Run forward
//		MXPredForward(predictor);
//
//		// Get output array from the class layer
//		const int outputLayerSize = 2;
//		const int outputLayerIndex = 0;
//		float *output = new float[outputLayerSize];
//		MXPredGetOutput(predictor, outputLayerIndex, &output[0], outputLayerSize);
//
//		// Load synsets
//		vector<string> synsets = loadSynsets("./src/net/model/mx-flow-shuffle/synset.txt");
//
//		// Print max index
//	/*  for (unsigned i=0; i<outputLayerSize; i++)
//		cout << output[i] << endl;
//	  cout << endl;
//	*/
//
//		int indmax = distance(output, max_element(output, output + outputLayerSize));
//		cout << indmax << synsets[indmax] << endl;
//
//
//		// Free memory
//		delete[] output;
//
//		MXPredFree(predictor);
//	}
//
//	return 0;
//}