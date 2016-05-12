#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CSVData.h"

struct ImageAnswer
{
	ImageAnswer();
	bool empty()
	{
		return image.empty();
	}
	cv::Mat image;
	std::string tseq, student, question;
	int score; //or string?
};

class ImageAnswerList
{
public:
	ImageAnswerList();
	void findAll(std::string datadir, std::string imgdir, std::string qid);
	ImageAnswer next();
	void setClip(cv::Range r, cv::Range c);
	void useDetail(const std::string &dethead);
	void rewind()
	{
		cur = 0;
	}
	bool atEnd() const
	{
		return cur >= files.size();
	}
private:
	std::string qid, pid, qord;
	std::string dethead;
	std::vector<std::string> files;
	int cur;
	CSVFile question, studentquestion, studentquestionsub;
	int scorescol, scoresdetailcol, sqidcol, sidcol;
	cv::Range rr, cr;
	ImageAnswer next0();
};
