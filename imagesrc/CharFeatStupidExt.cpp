#include "CharFeatStupidExt.h"

std::vector<double> CharFeatStupidExt::extractFeatures(const cv::Mat &sample)
{
	std::vector<double> ret;
	double vx, vy, vxy;
	stupid.variants(sample, vx, vy, vxy);
	double s = vx * vx + vy * vy + vxy * vxy * 2;
	//normalized covariant matrix
	ret.push_back(vx / s);
	ret.push_back(vy / s);
	ret.push_back(vxy / s);
	double sum = 0;
	for (int i = 0; i < sample.rows; i++)
		for (int j = 0; j < sample.cols; j++)
			sum += sample.at<float>(i, j);
	//blackness
	ret.push_back(sum / (sample.rows * sample.cols));
	//width-height ratio
	ret.push_back(sample.cols / (double)sample.rows);
	//width
	ret.push_back(sample.cols);
	//height
	ret.push_back(sample.rows);
	return ret;
}

std::vector<double> CharFeatStupidExt::extractSetFeatures(std::vector<cv::Mat> samples)
{
	std::vector<double> ret, t;
	for (const cv::Mat &sample : samples)
	{
		t = extractFeatures(sample);
		ret.resize(t.size(), 0);
		for (int i = 0; i < t.size(); i++)
			ret[i] += t[i];
	}
	for (int i = 0; i < ret.size(); i++)
		ret[i] /= samples.size();
	return ret;
}

std::vector<std::string> CharFeatStupidExt::getSetFeatureNames()
{
	std::vector<std::string> names;
	names.push_back("charvx");
	names.push_back("charvy");
	names.push_back("charvxy");
	names.push_back("charblack");
	names.push_back("whr");
	names.push_back("charw");
	names.push_back("charh");
	return names;
}
