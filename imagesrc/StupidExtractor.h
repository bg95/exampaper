#pragma once

#include "ImageAnswer.h"

class StupidExtractor
{
public:
	std::vector<double> extract(const ImageAnswer &a);
	std::vector<std::string> featureNames();
	void variants(const cv::Mat &img, double &vy, double &vx, double &vxy);
	void avgVar(const cv::Mat &img, double &avgx, double &avgy, double &vy, double &vx, double &vxy);
	double direction(const cv::Mat img);
};
