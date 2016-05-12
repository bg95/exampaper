#pragma once

#include "ImageAnswer.h"

struct Matrix
{
	Matrix(int row, int col)
		:rows(row), cols(col)
	{
		data.resize(row * col, 0.0f);
	}
	std::vector<float> data;
	int rows, cols;
	float &at(int i, int j)
	{
		return data[i * cols + j];
	}
	const float &at(int i, int j) const
	{
		return data[i * cols + j];
	}
};

template<class T>
struct Box
{
	Box()
	{}
	Box(T minx, T maxx, T miny, T maxy)
		:minx(minx), maxx(maxx), miny(miny), maxy(maxy)
	{}
	T minx, maxx, miny, maxy;
	T area() const
	{
		return rangeX() * rangeY();
	}
	T rangeX() const
	{
		return std::max(maxx - minx, (T)0);
	}
	T rangeY() const
	{
		return std::max(maxy - miny, (T)0);
	}
	cv::Rect toRect() const
	{
		return cv::Rect(miny, minx, rangeY(), rangeX());
	}
};
template<class T>
Box<T> intersect(const Box<T> &a, const Box<T> &b)
{
	return Box<T>(
			std::max(a.minx, b.minx),
			std::min(a.maxx, b.maxx),
			std::max(a.miny, b.miny),
			std::min(a.maxy, b.maxy));
}
template<class T>
Box<T> uni(const Box<T> &a, const Box<T> &b)
{
	return Box<T>(
			std::min(a.minx, b.minx),
			std::max(a.maxx, b.maxx),
			std::min(a.miny, b.miny),
			std::max(a.maxy, b.maxy));
}
template<class T>
Box<T> boundingBox(const std::vector<std::pair<T, T> > &points)
{
	Box<T> b(points[0].first, points[0].first, points[0].second, points[0].second);
	for (std::pair<T, T> p : points)
	{
		b.minx = std::min(b.minx, p.first);
		b.maxx = std::max(b.maxx, p.first);
		b.miny = std::min(b.miny, p.second);
		b.maxy = std::max(b.maxy, p.second);
	}
	return b;
}

template<class T>
T dist2(T x1, T y1, T x2, T y2)
{
	return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

std::vector<std::vector<std::pair<int, int> > > findChars(cv::Mat img);
std::vector<std::vector<std::pair<int, int> > > findCharsAnother(cv::Mat img, double tsx, double tsy);
std::vector<std::vector<std::pair<int, int> > > findCharsProjOnlyBestMerge(cv::Mat img, double tsx, double tsy);
std::vector<std::vector<std::pair<int, int> > > findCharsBestMerge(cv::Mat img);
//visit should be of type CV_32SC1 (int)
std::vector<std::pair<int, int> > floodfill(const cv::Mat &img, int i, int j, cv::Mat &visit, int stamp);
void allComponents(const cv::Mat &img, std::vector<std::vector<std::pair<int, int> > > &components, std::vector<Box<int> > &boundingboxes, int sizeth, double blackth, double blackthu);
std::vector<int> bestSplit(std::vector<double> hp, int minstep, int maxstep);
std::vector<std::pair<int, int> > bestMerge(const cv::Mat &img, std::vector<std::vector<std::pair<int, int> > > &components, std::vector<Box<int> > &boxes, double tsx, double tsy, double blankcost, double splitsegcost, double hspthl, double hspth, std::vector<int> &rank);
std::vector<std::pair<int, int> > bestMergeLine(std::vector<std::pair<Box<int>, int> > &boxind, double tsy);

cv::Mat toMat(std::vector<std::pair<int, int> > &points, Box<int> box);
Matrix toMatrix(std::vector<std::pair<int, int> > &points, Box<int> box);
void toMat(std::vector<std::pair<int, int> > points, Box<int> box, cv::Mat &img);
cv::Mat toMat(std::vector<float> v, Box<int> box);
std::vector<float> toStdVector(const cv::Mat &m);
std::vector<float> toStdVector(const Matrix &m);
