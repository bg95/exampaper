#pragma once

#include "StupidExtractor.h"

class CharFeatStupidExt
{
public:
	std::vector<double> extractFeatures(const cv::Mat &sample);
	std::vector<double> extractSetFeatures(std::vector<cv::Mat> samples);
	std::vector<std::string> getSetFeatureNames();
private:
	StupidExtractor stupid;
};
