#include "StupidExtractor.h"

std::vector<double> StupidExtractor::extract(const ImageAnswer &a)
{
	const cv::Mat &img = a.image;
	std::vector<double> ret;
	//average blackness
	double sum = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			sum += img.at<float>(i, j);
	ret.push_back(sum / (img.rows * img.cols));
	//average and standard deviation
	double avgx, avgy;
	double sdy, sdx, vxy;
	avgVar(img, avgx, avgy, sdx, sdy, vxy);
	sdy = std::sqrt(sdy);
	sdx = std::sqrt(sdx);
	ret.push_back(avgx);
	ret.push_back(avgy);
	ret.push_back(sdx);
	ret.push_back(sdy);
	//ret.push_back(vxy);
	return ret;
}

std::vector<std::string> StupidExtractor::featureNames()
{
	std::vector<std::string> names;
	names.push_back("avgblack");
	names.push_back("avgx");
	names.push_back("avgy");
	names.push_back("sdx");
	names.push_back("sdy");
	return names;
}

void StupidExtractor::variants(const cv::Mat &img, double &vx, double &vy, double &vxy)
{
	double avgx = 0, avgy = 0;
	avgVar(img, avgx, avgy, vx, vy, vxy);
}

void StupidExtractor::avgVar(const cv::Mat &img, double &avgx, double &avgy, double &vx, double &vy, double &vxy)
{
	avgx = 0, avgy = 0;
	vx = 0;
	vy = 0;
	vxy = 0;
	double sum = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			sum += img.at<float>(i, j);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			avgy += (img.at<float>(i, j)) * i;
			avgx += (img.at<float>(i, j)) * j;
		}
	avgy /= sum;
	avgx /= sum;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			vy += (img.at<float>(i, j)) * (i - avgy) * (i - avgy);
			vx += (img.at<float>(i, j)) * (j - avgx) * (j - avgx);
			vxy += (img.at<float>(i, j)) * (j - avgx) * (i - avgy);
		}
	vy /= sum;
	vx /= sum;
	vxy /= sum;
}
