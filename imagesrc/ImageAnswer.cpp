#include "ImageAnswer.h"

ImageAnswer::ImageAnswer()
	:image(0, 0, CV_32FC1)
{}

ImageAnswerList::ImageAnswerList()
{
	rr = cv::Range::all();
	cr = cv::Range::all();
}

void ImageAnswerList::findAll(std::string datadir, std::string imgdir, std::string qid)
{
	FILE *q;
	q = fopen((datadir + "tifen_question.csv").data(), "r");
	question.read(q, "ID"); //PAPER_ID, QUESTION_ORDER <- QUESTION_ID
	int pidcol = question.headerIndex("PAPER_ID");
	int qordcol = question.headerIndex("QUESTION_ORDER");
	int row = question.idIndex(qid);
	pid = question[row][pidcol];
	qord = question[row][qordcol];
	this->qid = qid;
	char cmd[1024];
	sprintf(cmd, "find %s | grep /%s.png > images_ImageAnswer.lst", imgdir.data(), qord.data());
	system(cmd);
	files.clear();
	FILE *f = fopen("images_ImageAnswer.lst", "r");
	while (!feof(f))
	{
		int c;
		std::string s;
		while ((c = getc(f)) != '\n')
		{
			if (c == -1)
				break;
			s.push_back(c);
		}
		if (!s.empty())
			files.push_back(s);
		fprintf(stderr, "Info: %d files found\n", (int)files.size());
	}
	cur = 0;

	FILE *sq = fopen((datadir + "tifen_studentquestion.csv").data(), "r");
	studentquestion.read(sq, {"TEMP_SEQUENCE", "PAPER_ID", "QUESTION_ORDER"}); //TEMP_SEQUENCE, PAPER_ID, QUESTION_ORDER -> SCORES
	sidcol = studentquestion.headerIndex("USER");
	scorescol = studentquestion.headerIndex("SCORES");
	sqidcol = studentquestion.headerIndex("ID");
	//scoresdetailcol = studentquestion.headerIndex("SCORES_DETAIL");

	//studentquestionsub
	FILE *sqs = fopen((datadir + "tifen_studentquestionsub.csv").data(), "r");
	studentquestionsub.read(sqs, "STUDENTQUESTION_ID"); // STUDENTQUESTION_ID (from studentquestion) -> SCORES_DETAIL
	scoresdetailcol = studentquestionsub.headerIndex("SCORES_DETAIL");
	fprintf(stderr, "Info: finding files finished, %d files found\n", (int)files.size());
}

void ImageAnswerList::setClip(cv::Range r, cv::Range c)
{
	rr = r;
	cr = c;
}

void ImageAnswerList::useDetail(const std::string &dethead)
{
	this->dethead = dethead;
}

ImageAnswer ImageAnswerList::next()
{
	ImageAnswer ia = next0();
	if (ia.image.rows >= rr.end && ia.image.cols >= cr.end)
		ia.image = ia.image(rr, cr);
	return ia;
}

ImageAnswer ImageAnswerList::next0()
{
	std::string fn;
	size_t last, secondlast, thirdlast;
	std::string tpid;
	ImageAnswer ret;
	do
	{
		if (atEnd())
			return ret;
		fn = files[cur];
		last = fn.find_last_of('/');
		if (last == std::string::npos)
		{
			cur++;
			continue;
		}
		secondlast = fn.find_last_of('/', last - 1);
		if (secondlast == std::string::npos)
		{
			cur++;
			continue;
		}
		thirdlast = fn.find_last_of('/', secondlast - 1);
		if (thirdlast == std::string::npos)
		{
			cur++;
			continue;
		}
		tpid = fn.substr(thirdlast + 1, secondlast - thirdlast - 1);
		if (tpid != pid)
		{
			cur++;
			continue;
		}
		cur++;
		ret.question = qid;
		ret.tseq = fn.substr(secondlast + 1, last - secondlast - 1);
		cv::Mat image = cv::imread(fn, 0/*grayscale*/);
		if (image.empty())
			fprintf(stderr, "Warning: cannot find file %s\n", fn.data());
		image.convertTo(ret.image, CV_32FC1);
		ret.image = cv::Mat::ones(ret.image.rows, ret.image.cols, CV_32FC1) - ret.image / 255.0;
		std::vector<std::string> sidpidqord;
		sidpidqord.push_back(ret.tseq);
		sidpidqord.push_back(pid);
		sidpidqord.push_back(qord);
		int scorei = studentquestion.idIndex(sidpidqord);
		if (scorei == -1)
		{
			fprintf(stderr, "Warning: cannot find %s,%s,%s\n", ret.tseq.data(), pid.data(), qord.data());
			ret.score = -1;
		}
		else
		{
			ret.student = studentquestion[scorei][sidcol];
			if (dethead.empty())
				ret.score = atoi(studentquestion[scorei][scorescol].data());
			else
			{
				std::string sqid = studentquestion[scorei][sqidcol];
				int sqidi = studentquestionsub.idIndex(sqid);
				if (sqidi == -1)
				{
					fprintf(stderr, "Warning: cannot find score of %s,%s,%s\n", ret.tseq.data(), pid.data(), qord.data());
					ret.score = -1;
					continue;
				}
				std::string det = studentquestionsub[sqidi][scoresdetailcol];
				size_t headpos = det.find(dethead);
				if (headpos == std::string::npos)
				{
					fprintf(stderr, "Warning: cannot find score of %s,%s,%s\n", ret.tseq.data(), pid.data(), qord.data());
					ret.score = -1;
					continue;
				}
				size_t numpos = det.find_first_of("0123456789", headpos + dethead.size());
				if (numpos == std::string::npos)
				{
					fprintf(stderr, "Warning: cannot find score of %s,%s,%s\n", ret.tseq.data(), pid.data(), qord.data());
					ret.score = -1;
					continue;
				}
				ret.score = atoi(det.data() + numpos);
			}
		}
	} while (false);
	return ret;
}
