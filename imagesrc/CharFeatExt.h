#pragma once

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class CharFeatExt
{
public:
	static const int PATCH_SIZE, CLUSTER_CNT;
	static const cv::Size CHAR_SIZE;

	CharFeatExt();
	void init(std::vector<cv::Mat> samples);
	std::vector<double> extractPatchFeatures(cv::Mat sample);
	std::vector<double> extractFeatures(cv::Mat sample);
	std::vector<double> extractSetFeatures(std::vector<cv::Mat> samples, int nbuck);
	std::vector<std::string> getSetFeatureNames(); //called AFTER extractSetFeatures
	void load(std::ifstream &is);
	void save(std::ofstream &os);

protected:
	std::vector<cv::Mat> kMeans(std::vector<cv::Mat> samples, int clusters);
	double distance(cv::Mat a, cv::Mat b);
	void normalize(std::vector<double> &v);

public:
	template<class T> static void saveStdVector(std::vector<T> &v, std::ostream &os)
	{
		os << " " << v.size();
		for (int i = 0; i < v.size(); i++)
			os << " " << v[i];
		os << "\n";
	}
	template<class T> static void loadStdVector(std::vector<T> &v, std::istream &is)
	{
		int n;
		is >> n;
		v.resize(n);
		for (int i = 0; i < n; i++)
			is >> v[i];
	}
	template<class T> static void append(std::vector<T> &a, const std::vector<T> &b)
	{
		for (const T &x : b)
			a.push_back(x);
	}

private:
	std::vector<cv::Mat> clusters;
	bool clusters_loaded;
	int nbuck;

	static void loadFloatMat(cv::Mat &m, std::istream &is);
	static void saveFloatMat(const cv::Mat &m, std::ostream &os);

};

