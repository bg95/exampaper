#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <map>
#include "ImageAnswer.h"
#include "StupidExtractor.h"
#include "ImageUtil.h"
#include "SVMModel.h"
#include "CharFeatExt.h"
#include "CharFeatStupidExt.h"
#include "Util.h"
#include "NormalizeChar.h"

double rms(std::vector<double> a, std::vector<double> b)
{
	double s = 0;
	for (int i = 0; i < a.size(); i++)
		s += (a[i] - b[i]) * (a[i] - b[i]);
	return std::sqrt(s / a.size());
}

cv::Mat normalizeChar(const cv::Mat &m, int w, int h)
{
	cv::Mat t(w, h, m.type());
	cv::resize(m, t, t.size());
	return t;
}

void dumpMat(int y, const Matrix &m, FILE *fp)
{
	fwrite(&y, sizeof(int), 1, fp);
	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++)
		{
			fwrite(&m.at(i, j), sizeof(float), 1, fp);
		}
}
void dumpMat(int y, const cv::Mat &m, FILE *fp)
{
	fwrite(&y, sizeof(int), 1, fp);
	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++)
		{
			fwrite(&m.at<float>(i, j), sizeof(float), 1, fp);
		}
}

void dumpMatVec(const std::vector<Matrix> &mv, FILE *fp)
{
	int n = mv.size();
	int m = mv[0].rows * mv[0].cols;
	fwrite(&n, sizeof(n), 1, fp);
	fwrite(&m, sizeof(m), 1, fp);
	for (const Matrix &mat: mv)
	{
		assert(mat.rows * mat.cols == m);
		//cv::imshow("showmat", mat);
		//cv::waitKey();
		dumpMat(0, mat, fp);
	}
}
void dumpMatVec(const std::vector<cv::Mat> &mv, FILE *fp)
{
	int n = mv.size();
	int m = mv[0].rows * mv[0].cols;
	fwrite(&n, sizeof(n), 1, fp);
	fwrite(&m, sizeof(m), 1, fp);
	for (const cv::Mat &mat: mv)
	{
		assert(mat.rows * mat.cols == m);
		//cv::imshow("showmat", mat);
		//cv::waitKey();
		dumpMat(0, mat, fp);
	}
}

void dumpMatVec(const std::vector<int> &y, const std::vector<Matrix> &mv, FILE *fp)
{
	int n = mv.size();
	int m = mv[0].rows * mv[0].cols;
	fwrite(&n, sizeof(n), 1, fp);
	fwrite(&m, sizeof(m), 1, fp);
	int i = 0;
	for (const Matrix &mat: mv)
	{
		assert(mat.rows * mat.cols == m);
		//cv::imshow("showmat", mat);
		//cv::waitKey();
		dumpMat(y[i], mat, fp);
		i++;
	}
}
void dumpMatVec(const std::vector<int> &y, const std::vector<cv::Mat> &mv, FILE *fp)
{
	int n = mv.size();
	int m = mv[0].rows * mv[0].cols;
	fwrite(&n, sizeof(n), 1, fp);
	fwrite(&m, sizeof(m), 1, fp);
	int i = 0;
	for (const cv::Mat &mat: mv)
	{
		assert(mat.rows * mat.cols == m);
		//cv::imshow("showmat", mat);
		//cv::waitKey();
		dumpMat(y[i], mat, fp);
		i++;
	}
}

void getMatVec(std::vector<cv::Mat> &mv, FILE *fp, int w, int h)
{
	int n, m;
	fread(&n, sizeof(n), 1, fp);
	fread(&m, sizeof(m), 1, fp);
	mv.resize(n);
	assert(m == w * h);
	for (int k = 0; k < n; k++)
	{
		float f;
		mv[k] = cv::Mat::zeros(w, h, CV_32FC1);
		for (int i = 0; i < mv[k].rows; i++)
			for (int j = 0; j < mv[k].cols; j++)
			{
				fread(&f, sizeof(f), 1, fp);
				mv[k].at<float>(i, j) = f;
			}
	}
}


void read_bin_data(FILE *fp, std::vector<std::vector<float> > &x, std::vector<int> &y)
{
	int n, m;
	int ty;
	float tx;
	fread(&n, sizeof(int), 1, fp);
	fread(&m, sizeof(int), 1, fp);
	printf("n = %d, m = %d\n", n, m);
	x.clear();
	y.clear();
	for (int i = 0; i < n; i++)
	{
		fread(&ty, sizeof(int), 1, fp);
		y.push_back(ty);
		x.push_back(std::vector<float>());
		for (int j = 0; j < m; j++)
		{
			fread(&tx, sizeof(float), 1, fp);
			x.back().push_back(tx);
		}
	}
}

void read_feature_data(FILE *fp, std::vector<std::vector<float> > &x)
{
	int n, m;
	float tx;
	fread(&n, sizeof(int), 1, fp);
	fread(&m, sizeof(int), 1, fp);
	x.clear();
	for (int i = 0; i < n; i++)
	{
		x.push_back(std::vector<float>());
		for (int j = 0; j < m; j++)
		{
			fread(&tx, sizeof(float), 1, fp);
			x.back().push_back(tx);
		}
	}
}

void write_feature_data(FILE *fp, const std::vector<std::vector<float> > &x)
{
	int n = x.size(), m = x[0].size();
	float tx;
	fwrite(&n, sizeof(int), 1, fp);
	fwrite(&m, sizeof(int), 1, fp);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			fwrite(&x[i][j], sizeof(float), 1, fp);
		}
	}
}

class Encoder
{
public:
	Encoder()
	{
		fifo_charmat = fopen("data/fifo_charmat", "wb");
		fifo_encoded = fopen("data/fifo_encoded", "rb");
	}
	~Encoder()
	{
		fclose(fifo_charmat);
		fclose(fifo_encoded);
	}
	Vector<float> encode(const cv::Mat m)
	{
		//encode
		int size = (m.rows * m.cols);
		fwrite(&size, sizeof(size), 1, fifo_charmat);
		for (int i = 0; i < m.rows; i++)
			for (int j = 0; j < m.cols; j++)
				fwrite(&m.at<float>(i, j), sizeof(float), 1, fifo_charmat);
		fflush(fifo_charmat);
		//fprintf(stderr, "data sent, waiting for reply\n");
		Vector<float> charvec;
		int charvecn;
		fread(&charvecn, sizeof(charvecn), 1, fifo_encoded);
		charvec.v.resize(charvecn);
		for (int i = 0; i < charvecn; i++)
			fread(&charvec.v[i], sizeof(charvec.v[i]), 1, fifo_encoded);
		//fprintf(stderr, "data received\n");
		return charvec;
	}
	Vector<float> encode_pair(const cv::Mat m, const cv::Mat m2)
	{
		//encode
		int size = (m.rows * m.cols) + (m2.rows * m2.cols);
		fwrite(&size, sizeof(size), 1, fifo_charmat);
		for (int i = 0; i < m.rows; i++)
			for (int j = 0; j < m.cols; j++)
				fwrite(&m.at<float>(i, j), sizeof(float), 1, fifo_charmat);
		for (int i = 0; i < m2.rows; i++)
			for (int j = 0; j < m2.cols; j++)
				fwrite(&m2.at<float>(i, j), sizeof(float), 1, fifo_charmat);
		fflush(fifo_charmat);
		//fprintf(stderr, "data sent, waiting for reply\n");
		Vector<float> charvec;
		int charvecn;
		fread(&charvecn, sizeof(charvecn), 1, fifo_encoded);
		charvec.v.resize(charvecn);
		for (int i = 0; i < charvecn; i++)
			fread(&charvec.v[i], sizeof(charvec.v[i]), 1, fifo_encoded);
		//fprintf(stderr, "data received\n");
		return charvec;
	}
	FILE *fifo_charmat;
	FILE *fifo_encoded;
};

void getAnswers(std::vector<ImageAnswerList> &answers_vec)
{
	answers_vec.resize(1);
	fprintf(stderr, "Info: finding files\n");
	answers_vec[0].findAll("../data/", "../imagedata/", "6d3631bf-ba4c-4d87-9291-3e16ae19629d"); //history question 29
	/*
	answers_vec[0].findAll("../data/", "../imagedata/", "40c221fe-5058-4a63-8541-65d54a3d103a"); //chinese question 1 (poems)
	answers_vec[0].setClip(cv::Range(546, 626), cv::Range(159, 1130));//second subquestion in 159,546 to 1130,626
	answers_vec[0].useDetail("\\\"1\\\":");//second subquestion \"1\"
	*/
	//answers_vec[0].findAll("../data/", "../imagedata/", "6d3631bf-ba4c-4d87-9291-3e16ae19629d"); //history question 29
	/*
	//answers_vec[1].findAll("../data/", "../imagedata/", "ce02b47d-8a8a-46c2-92f2-43f59e8f5921"); //history question 28
	answers_vec[2].findAll("../data/", "../imagedata/", "512038e9-eb32-49b9-a3c1-ba29eace4258"); //history question 26
	answers_vec[3].findAll("../data/", "../imagedata/", "5c8506f2-4ae8-4da9-bc3c-060b958f8a46"); //history question 27
	*/
	//answers.findAll("../data/", "../imagedata/", "90a70fad-70a7-4189-bd8e-39706b22f427");
	//answers.findAll("../data/", "../imagedata/", "40c221fe-5058-4a63-8541-65d54a3d103a");
}

int main_chars(std::string outputfilename)
{
	srand(time(0));
	std::vector<ImageAnswerList> answers_vec(1);
	getAnswers(answers_vec);
	//Extract Features Begin
	std::vector<cv::Mat> charsmat;
	for (ImageAnswerList &answers: answers_vec)
	{
		while (!answers.atEnd())
		{
			ImageAnswer ans = answers.next();
			if (ans.empty())
				continue;
			if (ans.score == -1)
				continue;
			std::vector<std::vector<std::pair<int, int> > > chars = findCharsProjOnlyBestMerge(ans.image, 30, 25);
			//std::vector<std::vector<std::pair<int, int> > > chars = findCharsBestMerge(ans.image);
			//std::vector<std::vector<std::pair<int, int> > > chars = findChars(ans.image);
			for (std::vector<std::pair<int, int> > c : chars)
			{
				Box<int> b = boundingBox(c);
				b.maxx += 1;
				b.maxy += 1;
				charsmat.push_back(toMat(c, b));
			}
		}
	}
	//write charsmat to a file
	for (cv::Mat &m : charsmat)
	{
		//normalize char
		//m = normalizeChar(m, 28, 28);
		m = normalizeCharNLN(m, 28, 28);
		/*
		cv::imshow("char", m);
		cv::waitKey();
		*/
	}
	/*
	FILE *fp = fopen("chardata_his29.bin", "wb");
	dumpMatVec(charsmat, fp);
	fclose(fp);
	*/
	std::vector<std::vector<float> > charsvec;
	for (const cv::Mat &m : charsmat)
		charsvec.push_back(toStdVector(m));
	FILE *fp = fopen(outputfilename.data(), "wb");
	write_feature_data(fp, charsvec);
	fclose(fp);
}

cv::Mat randomTranslate(cv::Mat a, int dx, int dy)
{
	cv::Mat b(a.rows, a.cols, a.type());
	dx = rand() % (2 * dx + 1) - dx;
	dy = rand() % (2 * dy + 1) - dy;
	for (int i = 0; i < b.rows; i++)
		for (int j = 0; j < b.cols; j++)
		{
			if (i + dx >= 0 && i + dx < a.rows && j + dy >= 0 && j + dy < a.cols)
				b.at<float>(i, j) = a.at<float>(i + dx, j + dy);
			else
				b.at<float>(i, j) = 0.0f;
		}
	return b;
}

int main_chars_translate()
{
	srand(time(0));
	std::vector<ImageAnswerList> answers_vec(1);
	//Extract Features Begin
	fprintf(stderr, "Info: finding files\n");
	answers_vec[0].findAll("../data/", "../imagedata/", "40c221fe-5058-4a63-8541-65d54a3d103a"); //chinese question 1 (poems)
	answers_vec[0].setClip(cv::Range(546, 626), cv::Range(159, 1130));//second subquestion in 159,546 to 1130,626
	answers_vec[0].useDetail("\\\"1\\\":");//second subquestion \"1\"
	//answers_vec[0].findAll("../data/", "../imagedata/", "6d3631bf-ba4c-4d87-9291-3e16ae19629d"); //history question 29
	/*
	answers_vec[1].findAll("../data/", "../imagedata/", "ce02b47d-8a8a-46c2-92f2-43f59e8f5921"); //history question 28
	answers_vec[2].findAll("../data/", "../imagedata/", "512038e9-eb32-49b9-a3c1-ba29eace4258"); //history question 26
	answers_vec[3].findAll("../data/", "../imagedata/", "5c8506f2-4ae8-4da9-bc3c-060b958f8a46"); //history question 27
	*/
	//answers.findAll("../data/", "../imagedata/", "90a70fad-70a7-4189-bd8e-39706b22f427");
	//answers.findAll("../data/", "../imagedata/", "40c221fe-5058-4a63-8541-65d54a3d103a");
	std::vector<cv::Mat> charsmat;
	for (ImageAnswerList &answers: answers_vec)
	{
		while (!answers.atEnd())
		{
			ImageAnswer ans = answers.next();
			if (ans.empty())
				continue;
			if (ans.score == -1)
				continue;
			//std::vector<std::vector<std::pair<int, int> > > chars = findCharsProjOnlyBestMerge(ans.image, 30, 25);
			std::vector<std::vector<std::pair<int, int> > > chars = findCharsBestMerge(ans.image);
			//std::vector<std::vector<std::pair<int, int> > > chars = findChars(ans.image);
			for (std::vector<std::pair<int, int> > c : chars)
			{
				Box<int> b = boundingBox(c);
				b.maxx += 1;
				b.maxy += 1;
				charsmat.push_back(toMat(c, b));
			}
		}
	}
	std::vector<cv::Mat> charsmat_translate;
	for (cv::Mat &m : charsmat)
	{
		//normalize char
		m = normalizeChar(m, 28, 28);
	}
	for (cv::Mat &m : charsmat)
	{
		for (int j = 0; j < 10; j++)
			charsmat_translate.push_back(randomTranslate(m, 3, 3));
	}
	/*
	FILE *fp = fopen("chardata_his29.bin", "wb");
	dumpMatVec(charsmat, fp);
	fclose(fp);
	*/
	//write charsmat to a file
	std::vector<std::vector<float> > charsvec;
	for (const cv::Mat &m : charsmat_translate)
		charsvec.push_back(toStdVector(m));
	FILE *fp = fopen("data/chardata_chn1.2_translate.bin", "wb");
	write_feature_data(fp, charsvec);
	fclose(fp);
}

int main_chars_pipe()
{
	srand(time(0));
	std::vector<ImageAnswerList> answers_vec(1);
	//Extract Features Begin
	fprintf(stderr, "Info: finding files\n");
	answers_vec[0].findAll("../data/", "../imagedata/", "6d3631bf-ba4c-4d87-9291-3e16ae19629d"); //history question 29
	//answers_vec[0].findAll("../data/", "../imagedata/", "40c221fe-5058-4a63-8541-65d54a3d103a"); //chinese question 1 (poems)
	/*
	answers_vec[1].findAll("../data/", "../imagedata/", "ce02b47d-8a8a-46c2-92f2-43f59e8f5921"); //history question 28
	answers_vec[2].findAll("../data/", "../imagedata/", "512038e9-eb32-49b9-a3c1-ba29eace4258"); //history question 26
	answers_vec[3].findAll("../data/", "../imagedata/", "5c8506f2-4ae8-4da9-bc3c-060b958f8a46"); //history question 27
	*/
	//answers.findAll("../data/", "../imagedata/", "90a70fad-70a7-4189-bd8e-39706b22f427");
	//answers.findAll("../data/", "../imagedata/", "40c221fe-5058-4a63-8541-65d54a3d103a");
	for (ImageAnswerList &answers: answers_vec)
	{
		while (!answers.atEnd())
		{
			std::vector<cv::Mat> charsmat; //count for every image
			ImageAnswer ans = answers.next();
			if (ans.empty())
				continue;
			if (ans.score == -1)
				continue;
			//std::vector<std::vector<std::pair<int, int> > > chars = findCharsBestMerge(ans.image);
			std::vector<std::vector<std::pair<int, int> > > chars = findCharsProjOnlyBestMerge(ans.image, 30, 25); //typical size 40x50
			//std::vector<std::vector<std::pair<int, int> > > chars = findChars(ans.image);
			for (std::vector<std::pair<int, int> > c : chars)
			{
				Box<int> b = boundingBox(c);
				b.maxx += 1;
				b.maxy += 1;
				charsmat.push_back(toMat(c, b));
			}
			//write charsmat to stdout
			float score = ans.score;
			fwrite(&score, sizeof(float), 1, stdout);
			int n = charsmat.size();
			fwrite(&n, sizeof(int), 1, stdout);
			for (cv::Mat &m : charsmat)
			{
				//normalize char
				m = normalizeChar(m, 28, 28);
				m = normalizeCharNLN(m, 28, 28);
				for (int i = 0; i < m.rows; i++)
					for (int j = 0; j < m.cols; j++)
					{
						fwrite(&m.at<float>(i, j), sizeof(float), 1, stdout);
					}
			}
		}
	}
	//end of output
	int n = -1;
	fwrite(&n, sizeof(int), 1, stdout);
}

int main_chars_encode(std::string outputfile)
{
	srand(time(0));
	std::vector<ImageAnswerList> answers_vec(1);
	getAnswers(answers_vec);
	//Extract Features Begin
	Encoder encoder;
	std::vector<std::vector<float> > features;
	//std::vector<std::vector<float> > allchars;
	for (ImageAnswerList &answers: answers_vec)
	{
		while (!answers.atEnd())
		{
			std::vector<cv::Mat> charsmat; //count for every image
			ImageAnswer ans = answers.next();
			if (ans.empty())
				continue;
			if (ans.score == -1)
				continue;
			std::vector<std::vector<std::pair<int, int> > > chars = findCharsProjOnlyBestMerge(ans.image, 30, 25);
			//std::vector<std::vector<std::pair<int, int> > > chars = findCharsBestMerge(ans.image);
			//std::vector<std::vector<std::pair<int, int> > > chars = findChars(ans.image);
			for (std::vector<std::pair<int, int> > c : chars)
			{
				Box<int> b = boundingBox(c);
				b.maxx += 1;
				b.maxy += 1;
				charsmat.push_back(toMat(c, b));
			}
			float score = ans.score;
			//std::vector<Vector<float> > encoded;
			for (cv::Mat &m : charsmat)
			{
				//m = normalizeChar(m, 28, 28);
				m = normalizeCharNLN(m, 28, 28);
				//encoded.push_back(encoder.encode(m));
			}
			for (int i = 0; i < charsmat.size(); i++)
			{
				features.push_back(encoder.encode(charsmat[i]).v);
				//allchars.push_back(toStdVector(charsmat[i]));
			}
			printf(".");
		}
		printf("\n");
	}
	
	FILE *fpout = fopen(outputfile.data(), "wb");
	write_feature_data(fpout, features);
	fclose(fpout);
	/*
	FILE *fpout = fopen("data/char_pair_data.bin", "wb");
	write_feature_data(fpout, charpairs);
	fclose(fpout);
	*/
}

int main_char_pairs_encode()
{
	srand(time(0));
	std::vector<ImageAnswerList> answers_vec(4);
	//Extract Features Begin
	fprintf(stderr, "Info: finding files\n");
	answers_vec[0].findAll("../data/", "../imagedata/", "6d3631bf-ba4c-4d87-9291-3e16ae19629d"); //history question 29
	
	answers_vec[1].findAll("../data/", "../imagedata/", "ce02b47d-8a8a-46c2-92f2-43f59e8f5921"); //history question 28
	answers_vec[2].findAll("../data/", "../imagedata/", "512038e9-eb32-49b9-a3c1-ba29eace4258"); //history question 26
	answers_vec[3].findAll("../data/", "../imagedata/", "5c8506f2-4ae8-4da9-bc3c-060b958f8a46"); //history question 27
	
	//answers.findAll("../data/", "../imagedata/", "90a70fad-70a7-4189-bd8e-39706b22f427");
	//answers.findAll("../data/", "../imagedata/", "40c221fe-5058-4a63-8541-65d54a3d103a");
	Encoder encoder;
	std::vector<std::vector<float> > features;
	std::vector<std::vector<float> > charpairs;
	for (ImageAnswerList &answers: answers_vec)
	{
		while (!answers.atEnd())
		{
			std::vector<cv::Mat> charsmat; //count for every image
			ImageAnswer ans = answers.next();
			if (ans.empty())
				continue;
			if (ans.score == -1)
				continue;
			std::vector<std::vector<std::pair<int, int> > > chars = findCharsBestMerge(ans.image);
			//std::vector<std::vector<std::pair<int, int> > > chars = findChars(ans.image);
			for (std::vector<std::pair<int, int> > c : chars)
			{
				Box<int> b = boundingBox(c);
				b.maxx += 1;
				b.maxy += 1;
				charsmat.push_back(toMat(c, b));
			}
			float score = ans.score;
			//std::vector<Vector<float> > encoded;
			for (cv::Mat &m : charsmat)
			{
				m = normalizeChar(m, 28, 28);
				//encoded.push_back(encoder.encode(m));
			}
			/*
			for (int i = 1; i < encoded.size(); i++)
			{
				features.push_back(std::vector<float>());
				for (float x : encoded[i - 1].v)
					features.back().push_back(x);
				for (float x : encoded[i].v)
					features.back().push_back(x);
			}
			for (int i = 1; i < charsmat.size(); i++)
			{
				charpairs.push_back(std::vector<float>());
				for (cv::MatConstIterator_<float> it = charsmat[i - 1].begin<float>(); it != charsmat[i - 1].end<float>(); it++)
					charpairs.back().push_back(*it);
				for (cv::MatConstIterator_<float> it = charsmat[i].begin<float>(); it != charsmat[i].end<float>(); it++)
					charpairs.back().push_back(*it);
			}*/
			for (int i = 1; i < charsmat.size(); i++)
			{
				features.push_back(encoder.encode_pair(charsmat[i - 1], charsmat[i]).v);
			}
			printf(".");
		}
		printf("\n");
	}
	
	FILE *fpout = fopen("data/encoded_features_char_pair_data_pair_encoder", "wb");
	write_feature_data(fpout, features);
	fclose(fpout);
	/*
	FILE *fpout = fopen("data/char_pair_data.bin", "wb");
	write_feature_data(fpout, charpairs);
	fclose(fpout);
	*/
}

int main_chars_mark()
{
	srand(time(0));
	double prob = 0.02;
	std::vector<ImageAnswerList> answers_vec(1);
	//Extract Features Begin
	fprintf(stderr, "Info: finding files\n");
	answers_vec[0].findAll("../data/", "../imagedata/", "6d3631bf-ba4c-4d87-9291-3e16ae19629d"); //history question 29
	/*
	answers_vec[1].findAll("../data/", "../imagedata/", "ce02b47d-8a8a-46c2-92f2-43f59e8f5921"); //history question 28
	answers_vec[2].findAll("../data/", "../imagedata/", "512038e9-eb32-49b9-a3c1-ba29eace4258"); //history question 26
	answers_vec[3].findAll("../data/", "../imagedata/", "5c8506f2-4ae8-4da9-bc3c-060b958f8a46"); //history question 27
	*/
	//answers.findAll("../data/", "../imagedata/", "90a70fad-70a7-4189-bd8e-39706b22f427");
	//answers.findAll("../data/", "../imagedata/", "40c221fe-5058-4a63-8541-65d54a3d103a");
	std::vector<cv::Mat> charsmat;
	std::vector<int> y;
	std::unordered_map<std::string, int> namemap;
	std::vector<std::string> namemapinv;
	namemap.emplace("", (int)namemap.size());
	namemapinv.resize(namemap.size());
	namemapinv[namemap[""]] = "";

	int cur = 0;

	for (ImageAnswerList &answers: answers_vec)
	{
		while (!answers.atEnd())
		{
			ImageAnswer ans = answers.next();
			if (ans.empty())
				continue;
			if (ans.score == -1)
				continue;
			std::vector<std::vector<std::pair<int, int> > > chars = findCharsBestMerge(ans.image);
			//std::vector<std::vector<std::pair<int, int> > > chars = findChars(ans.image);
			for (std::vector<std::pair<int, int> > c : chars)
			{
				if (rand() < RAND_MAX * prob)
				{
					Box<int> b = boundingBox(c);
					b.maxx += 1;
					b.maxy += 1;
					cv::Mat img = ans.image;
					cv::Mat rectimg;
					img.convertTo(rectimg, CV_32FC3);
					cv::resize(rectimg, rectimg, cv::Size(0, 0), 0.5, 0.5);
					cv::rectangle(rectimg, cv::Rect(b.miny / 2, b.minx / 2, b.rangeY() / 2, b.rangeX() / 2) , cv::Scalar(1, 0, 0));
					cv::imshow("mat to mark", rectimg);
					cv::waitKey(100);
					std::cout << "Char #" << cur << "name: ";
					std::string name;
					//std::cin >> name;
					std::getline(std::cin, name);
					if (name == std::string("prev"))
					{
						cur--;
						continue;
					}
					namemap.emplace(name, (int)namemap.size());
					namemapinv.resize(namemap.size());
					namemapinv[namemap[name]] = name;
					charsmat.push_back(toMat(c, b));
					y.push_back(namemap[name]);
				}
				cur++;
			}
		}
	}
	//write charsmat to a file
	for (cv::Mat &m : charsmat)
	{
		//normalize char
		m = normalizeChar(m, 28, 28);
	}
	FILE *fp = fopen("chardata_his29_marked.bin", "wb");
	dumpMatVec(y, charsmat, fp);
	fclose(fp);

	fp = fopen("namemap", "w");
	for (std::string str : namemapinv)
		fprintf(fp, "%s, %d\n", str.data(), namemap[str]);
	fclose(fp);
	return 0;
}

int main_mark()
{
	std::vector<cv::Mat> charsmat, markedcharsmat;
	std::vector<int> y;
	std::unordered_map<std::string, int> namemap;
	std::vector<std::string> namemapinv;
	FILE *fp = fopen("chardata_his29_findchars.bin", "rb");
	getMatVec(charsmat, fp, 28, 28);
	fclose(fp);

	for (cv::Mat m : charsmat)
		if (rand() < RAND_MAX /* 0.02*/)
			markedcharsmat.push_back(m);
	y.resize(markedcharsmat.size(), 0);
	namemap.emplace("", (int)namemap.size());
	namemapinv.resize(namemap.size());
	namemapinv[namemap[""]] = "";
	printf("%d mats selected\n", (int)markedcharsmat.size());
	int cur;
	cur = 0;
	while (cur < markedcharsmat.size())
	{
		cv::Mat m = markedcharsmat[cur];
		cv::Mat mag(280, 280, CV_32FC1);
		cv::resize(m, mag, mag.size());
		cv::imshow("mat to mark", mag);
		cv::waitKey(100);
		std::cout << "Char #" << cur << "(" << namemapinv[y[cur]] << ") new name: ";
		std::string name;
		//std::cin >> name;
		std::getline(std::cin, name);
		if (name == std::string("prev"))
		{
			cur--;
			continue;
		}
		namemap.emplace(name, (int)namemap.size());
		namemapinv.resize(namemap.size());
		namemapinv[namemap[name]] = name;
		cur++;
	}

	fp = fopen("chardata_his29_marked.bin", "wb");
	dumpMatVec(y, markedcharsmat, fp);
	fclose(fp);

	return 0;
}

int main0()
{
	srand(time(0));
	ImageAnswerList answers;
	StupidExtractor stupid;
	std::vector<std::vector<double> > x, x1;
	std::vector<std::string> xname;
	std::vector<double> y;
	SVMModel svm, svm1;
	//Extract Features Begin
	if (true)
	{
	fprintf(stderr, "Info: finding files\n");
	//answers.findAll("../data/", "../imagedata/", "6d3631bf-ba4c-4d87-9291-3e16ae19629d");
	//answers.findAll("../data/", "../imagedata/", "ce02b47d-8a8a-46c2-92f2-43f59e8f5921");
	answers.findAll("../data/", "../imagedata/", "90a70fad-70a7-4189-bd8e-39706b22f427");
	//answers.findAll("../data/", "../imagedata/", "40c221fe-5058-4a63-8541-65d54a3d103a");
	int cnt = 0;
	//std::vector<cv::Mat> charsmat;
	CharFeatStupidExt cfse;
	CharFeatExt cfe;
	{
		std::ifstream cif("clusters");
		cfe.load(cif);
	}
	while (!answers.atEnd())
	{
		ImageAnswer ans = answers.next();
		if (ans.empty())
			continue;
		if (ans.score == -1)
			continue;
		y.push_back(ans.score);
		x.push_back(stupid.extract(ans)); //stupid
		xname = stupid.featureNames();
		x1.push_back(std::vector<double>(1, x.back()[0])); //x1 has only blackness
		cnt++;
		fprintf(stderr, "Info: extracted %d\n", cnt);
		//std::vector<std::vector<std::pair<int, int> > > chars = findChars(ans.image);
		//std::vector<std::vector<std::pair<int, int> > > chars = findCharsAnother(ans.image, 28, 28);
		std::vector<std::vector<std::pair<int, int> > > chars = findCharsBestMerge(ans.image);
		std::vector<cv::Mat> charsmat;
		for (std::vector<std::pair<int, int> > c : chars)
		{
			Box<int> b = boundingBox(c);
			b.maxx += 1;
			b.maxy += 1;
			charsmat.push_back(toMat(c, b));
		}
		x.back().push_back(chars.size()); //# chars
		xname.push_back("numchars");

		/*
		std::vector<double> charsetfeat = cfe.extractSetFeatures(charsmat, 1);
		std::vector<std::string> charsetfeatname = cfe.getSetFeatureNames();
		for (double csf : charsetfeat)
			x.back().push_back(csf); //char set features
		for (std::string csfn : charsetfeatname)
			xname.push_back(csfn); //char set feature names
			*/

		std::vector<double> charsetstupidfeat = cfse.extractSetFeatures(charsmat);
		std::vector<std::string> charsetstupidfeatname = cfse.getSetFeatureNames();
		for (double cssf : charsetstupidfeat)
			x.back().push_back(cssf); //char set stupid features
		for (std::string cssfn : charsetstupidfeatname)
			xname.push_back(cssfn); //char set stupid feature names
	}
	/*
	cfe.init(charsmat);
	{
		std::ofstream cof("clusters");
		cfe.save(cof);
	}
	*/
	FILE *featout = fopen("features_char_8e6b_08.csv", "w");
	fprintf(featout, "y");
	for (int j = 0; j < x[0].size(); j++)
		fprintf(featout, ",%s", xname[j].data());
	fprintf(featout, "\n");
	for (int i = 0; i < x.size(); i++)
		//if (y[i] > 0.001)
	{
		fprintf(featout, "%lf", y[i]);
		for (int j = 0; j < x[i].size(); j++)
			fprintf(featout, ",%lf", x[i][j]);
		fprintf(featout, "\n");
	}
	fclose(featout);
	printf("y");
	for (int j = 0; j < x[0].size(); j++)
		printf(",%s", xname[j].data());
	printf("\n");
	for (int i = 0; i < x.size(); i++)
	{
		printf("%lf", y[i]);
		for (int j = 0; j < x[i].size(); j++)
			printf(",%lf", x[i][j]);
		printf("\n");
	}
	} //Extract Features End


	return 0;

	const double testpart = 0.1;
	std::vector<std::vector<double> > trainx, testx;
	std::vector<std::vector<double> > trainx1, testx1;
	std::vector<double> trainy, testy, resy;
	std::vector<double> resy1;
	double avgy = 0;
	for (int i = 0; i < x.size(); i++)
	{
		//if (i < x.size() * testpart)
		if (rand() < RAND_MAX * testpart)
		{
			testx.push_back(x[i]);
			testx1.push_back(x1[i]);
			testy.push_back(y[i]);
		}
		else
		{
			trainx.push_back(x[i]);
			trainx1.push_back(x1[i]);
			trainy.push_back(y[i]);
			avgy += y[i];
		}
	}
	avgy /= trainy.size();
	svm.train(trainx, trainy);
	svm1.train(trainx1, trainy);
	svm.save("svmmodel");
	svm.load("svmmodel");
	for (int i = 0; i < testy.size(); i++)
		resy.push_back(svm.predict(testx[i]));
	for (int i = 0; i < testy.size(); i++)
		resy1.push_back(svm1.predict(testx1[i]));
	//resy = svm.predict(testx);
	for (int i = 0; i < testy.size(); i++)
		printf("%lf %lf %lf %lf\n", testy[i], resy[i], resy1[i], avgy);
	printf("rms = %lf, rms1 = %lf, avg err = %lf\n", rms(testy, resy), rms(testy, resy1), rms(testy, std::vector<double>(testy.size(), avgy)));
	return 0;
}

int main_kmeans(std::string featurefilename, std::string clusterfilename, int numclusters, double portion = 1.0)
{
	srand(time(0));
	//load features
	std::vector<std::vector<float> > f;
	//FILE *featurefile = fopen("data/encoded_features_char_pair_data_pair_encoder", "rb");
	FILE *featurefile = fopen(featurefilename.data(), "rb");
	read_feature_data(featurefile, f);
	fclose(featurefile);
	//clustering
	//double portion = 1;
	//int numclusters = 300;
	std::vector<Vector<float> > fv;
	std::vector<int> fvpos;
	for (int i = 0; i < f.size(); i++)
		if (rand() < RAND_MAX * portion)
		{
			fv.push_back(Vector<float>(f[i]));
			fvpos.push_back(i);
		}
	std::vector<int> cluster;
	std::vector<Vector<float> > centers;
	centers = kMeans(fv, numclusters, cluster);
	//save clusters
	//FILE *clustersfile = fopen("data/encoded_features_char_pair_data_pair_encoder_0.15_300_clusters", "wb");
	FILE *clustersfile = fopen(clusterfilename.data(), "wb");
	int n = cluster.size();
	fwrite(&n, sizeof(n), 1, clustersfile);
	for (int v : fvpos)
		fwrite(&v, sizeof(v), 1, clustersfile);
	for (int v : cluster)
		fwrite(&v, sizeof(v), 1, clustersfile);
	fclose(clustersfile);
}
void displayChars(const std::vector<std::vector<float> > &chars)
{
	const int stepx = 30, stepy = 30;
	const int charx = 28, chary = 28;
	int n = chars.size();
	if (n == 0)
	{
		fprintf(stderr, "Nothing to display!\n");
		cv::Mat disp = cv::Mat::zeros(30, 30, CV_32FC1);
		cv::imshow("chars", disp);
		return;
	}
	int r = std::sqrt(n);
	int c = (n + r - 1) / r;
	cv::Mat disp = cv::Mat::zeros(r * stepx, c * stepy, CV_32FC1);
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c && i * c + j < n; j++)
		{
			if (chars[i * c + j].size() != charx * chary)
				continue;
			for (int ti = 0; ti < charx; ti++)
				for (int tj = 0; tj < chary; tj++)
					disp.at<float>(i * stepx + ti, j * stepy + tj) = chars[i * c + j][ti * chary + tj];
		}
	cv::imshow("chars", disp);
}
int main_kmeans_display_char_pair() //char pair ver
{
	//load dataset
	/*
	std::vector<std::vector<float> > x;
	std::vector<int> y;
	FILE *datasetfile = fopen("chardata.bin", "r");
	read_bin_data(datasetfile, x, y);
	fclose(datasetfile);
	*/
	std::vector<std::vector<float> > x;
	FILE *datasetfile = fopen("data/char_pair_data.bin", "r");
	read_feature_data(datasetfile, x);
	fclose(datasetfile);
	//load clusters
	std::vector<int> cluster;
	std::vector<int> fvpos;
	int n;
	FILE *clustersfile = fopen("data/encoded_features_char_pair_data_pair_encoder_0.15_300_clusters", "r");
	fread(&n, sizeof(n), 1, clustersfile);
	cluster.resize(n);
	fvpos.resize(n);
	for (int &v : fvpos)
		fread(&v, sizeof(v), 1, clustersfile);
	for (int &v : cluster)
		fread(&v, sizeof(v), 1, clustersfile);
	fclose(clustersfile);
	//display
	int ncluster = 0;
	for (int v : cluster)
		if (ncluster < v)
			ncluster = v;
	ncluster++;
	for (int c = 0; c < ncluster; c++)
	{
		std::vector<std::vector<float> > chars;
		for (int i = 0; i < n; i++)
			if (cluster[i] == c)
			{
				chars.push_back(x[fvpos[i]]);
			}
		std::vector<std::vector<float> > splitchars;
		for (std::vector<float> ch : chars)
		{
			std::vector<float> first, second;
			first = ch;
			first.resize(first.size() / 2);
			for (int i = first.size(); i < ch.size(); i++)
				second.push_back(ch[i]);
			splitchars.push_back(first);
			splitchars.push_back(second);
		}
		displayChars(splitchars);
		cv::waitKey();
	}
}
int main_kmeans_display(std::string datasetfilename, std::string clusterfilename) //char ver
{
	//load dataset
	/*
	std::vector<std::vector<float> > x;
	std::vector<int> y;
	FILE *datasetfile = fopen("chardata.bin", "r");
	read_bin_data(datasetfile, x, y);
	fclose(datasetfile);
	*/
	std::vector<std::vector<float> > x;
	FILE *datasetfile = fopen(datasetfilename.data(), "r");
	read_feature_data(datasetfile, x);
	fclose(datasetfile);
	//load clusters
	std::vector<int> cluster;
	std::vector<int> fvpos;
	int n;
	FILE *clustersfile = fopen(clusterfilename.data(), "r");
	fread(&n, sizeof(n), 1, clustersfile);
	cluster.resize(n);
	fvpos.resize(n);
	for (int &v : fvpos)
		fread(&v, sizeof(v), 1, clustersfile);
	for (int &v : cluster)
		fread(&v, sizeof(v), 1, clustersfile);
	fclose(clustersfile);
	//display
	int ncluster = 0;
	for (int v : cluster)
		if (ncluster < v)
			ncluster = v;
	ncluster++;
	for (int c = 0; c < ncluster; c++)
	{
		std::vector<std::vector<float> > chars;
		for (int i = 0; i < n; i++)
			if (cluster[i] == c)
			{
				chars.push_back(x[fvpos[i]]);
			}
		displayChars(chars);
		cv::waitKey();
		// normalization
		for (int i = 0; i < n; i++)
			if (cluster[i] == c)
			{
				chars.push_back(toStdVector(normalizeCharNLN(toMat(x[fvpos[i]], Box<int>(0, 28, 0, 28)), 28, 28)));
			}
		displayChars(chars);
		cv::waitKey();
	}
}
int main_kmeans_feature(const char *featurefilename, const char *datasetfilename, const char *clustersfilename)
{
	srand(time(0));
	//load features
	fprintf(stderr, "Info: loading features\n");
	std::vector<std::vector<float> > f;
	FILE *featurefile = fopen(featurefilename, "r");
	read_feature_data(featurefile, f);
	fclose(featurefile);
	//load dataset
	fprintf(stderr, "Info: loading chardata\n");
	std::vector<std::vector<float> > x;
	std::vector<int> y;
	FILE *datasetfile = fopen(datasetfilename, "r");
	read_bin_data(datasetfile, x, y);
	fclose(datasetfile);
	//load clusters
	fprintf(stderr, "Info: loading clusters\n");
	std::vector<int> cluster;
	std::vector<int> clustersizes;
	std::vector<int> fvpos;
	std::vector<Vector<float> > centers;
	int n;
	FILE *clustersfile = fopen(clustersfilename, "r");
	fread(&n, sizeof(n), 1, clustersfile);
	cluster.resize(n);
	fvpos.resize(n);
	for (int &v : fvpos)
		fread(&v, sizeof(v), 1, clustersfile);
	for (int &v : cluster)
		fread(&v, sizeof(v), 1, clustersfile);
	fclose(clustersfile);
	//calculate centers
	fprintf(stderr, "Info: calculate centers\n");
	int ncluster = 0;
	for (int v : cluster)
		if (ncluster < v)
			ncluster = v;
	ncluster++;
	centers.resize(ncluster, std::vector<float>(f[0].size(), 0));
	clustersizes.resize(ncluster, 0);
	for (int i = 0; i < n; i++)
	{
		centers[cluster[i]] += Vector<float>(f[fvpos[i]]);
		clustersizes[cluster[i]]++;
	}
	for (int c = 0; c < ncluster; c++)
		centers[c] *= 1.0 / clustersizes[c];
	//access fifos
	//start encoder
	fprintf(stderr, "Info: start encoder\n");
	Encoder encoder;
	//Extract Features Begin
	std::vector<std::vector<double> > features; //for SVM
	std::vector<double> targets; //for SVM
	fprintf(stderr, "Info: extract features begin\n");
	std::vector<ImageAnswerList> answers_vec(1);
	getAnswers(answers_vec);
	for (ImageAnswerList &answers: answers_vec)
	{
		while (!answers.atEnd())
		{
			std::vector<cv::Mat> charsmat; //count for every image
			ImageAnswer ans = answers.next();
			if (ans.empty())
				continue;
			if (ans.score == -1)
				continue;
			std::vector<std::vector<std::pair<int, int> > > chars = findCharsProjOnlyBestMerge(ans.image, 30, 25);
			//std::vector<std::vector<std::pair<int, int> > > chars = findCharsBestMerge(ans.image);
			//std::vector<std::vector<std::pair<int, int> > > chars = findChars(ans.image);
			for (std::vector<std::pair<int, int> > c : chars)
			{
				Box<int> b = boundingBox(c);
				b.maxx += 1;
				b.maxy += 1;
				charsmat.push_back(toMat(c, b));
			}
			float score = ans.score;
			std::vector<int> charclustercnt(ncluster, 0);
			for (cv::Mat &m : charsmat)
			{
				//normalize char
				m = normalizeCharNLN(m, 28, 28);
				//encode
				Vector<float> charvec;
				charvec = encoder.encode(m);
				//find nearest center (mindiscntr)
				double mindis, tdis;
				int mindiscntr, j;
				mindis = 1.0 / 0.0; //Infinity
				for (j = 0; j < ncluster; j++)
				{
					tdis = distance(charvec, centers[j]);
					if (tdis < mindis)
					{
						mindis = tdis;
						mindiscntr = j;
					}
				}
				charclustercnt[mindiscntr]++;
			}//for charsmat
			//output
			printf("%f", score);
			for (int i = 0; i < charclustercnt.size(); i++)
				printf(",%d", charclustercnt[i]);
			printf("\n");
			fprintf(stderr, ".");
			targets.push_back(score);
			std::vector<double> charclustercntdouble;
			for (int i = 0; i < charclustercnt.size(); i++)
				charclustercntdouble.push_back(charclustercnt[i]);
			features.push_back(charclustercntdouble);
		}//while each answer
	}

	//train using SVM
	std::vector<std::vector<double> > features_train; //for SVM
	std::vector<double> targets_train; //for SVM
	std::vector<std::vector<double> > features_test; //for SVM
	std::vector<double> targets_test; //for SVM
	int test_cnt = targets.size() / 10;
	for (int i = 0; i < targets.size() - test_cnt; i++)
	{
		features_train.push_back(features[i]);
		targets_train.push_back(targets[i]);
	}
	for (int i = targets.size() - test_cnt; i < targets.size(); i++)
	{
		features_test.push_back(features[i]);
		targets_test.push_back(targets[i]);
	}
	SVMModel svm;
	svm.train(features_train, targets_train);
	std::vector<double> y_predict = svm.predict(features_test);
	double err = rms(targets_test, y_predict);
	printf("err = %lf\n", err);
}

int main_chars_automark()
{
	srand(time(0));
	std::vector<ImageAnswerList> answers_vec(1);
	//Extract Features Begin
	fprintf(stderr, "Info: finding files\n");
	answers_vec[0].findAll("../data/", "../imagedata/", "40c221fe-5058-4a63-8541-65d54a3d103a"); //chinese question 1 (poems)
	answers_vec[0].setClip(cv::Range(546, 626), cv::Range(159, 1130));//second subquestion in 159,546 to 1130,626
	answers_vec[0].useDetail("\\\"1\\\":");//second subquestion \"1\"
	/*
	answers_vec[0].findAll("../data/", "../imagedata/", "6d3631bf-ba4c-4d87-9291-3e16ae19629d"); //history question 29
	
	answers_vec[1].findAll("../data/", "../imagedata/", "ce02b47d-8a8a-46c2-92f2-43f59e8f5921"); //history question 28
	answers_vec[2].findAll("../data/", "../imagedata/", "512038e9-eb32-49b9-a3c1-ba29eace4258"); //history question 26
	answers_vec[3].findAll("../data/", "../imagedata/", "5c8506f2-4ae8-4da9-bc3c-060b958f8a46"); //history question 27
	*/
	
	//answers.findAll("../data/", "../imagedata/", "90a70fad-70a7-4189-bd8e-39706b22f427");
	//answers.findAll("../data/", "../imagedata/", "40c221fe-5058-4a63-8541-65d54a3d103a");
	std::vector<std::vector<std::pair<int, int> > > allchars;
	std::vector<cv::Mat> charsmat;
	std::vector<int> y;

	int charposx[] = {585, 585, 585, 585, 585, 585, 585, 585, 585,  585,  585,  585};
	int charposy[] = {220, 300, 468, 559, 706, 744, 784, 822, 975, 1015, 1055, 1095};
	int ordcnt = sizeof(charposx) / sizeof(charposx[0]);

	for (ImageAnswerList &answers: answers_vec)
	{
		while (!answers.atEnd())
		{
			ImageAnswer ans = answers.next();
			if (ans.empty())
				continue;
			if (ans.score == -1)
				continue;
			if (ans.score < 4) //ignore not full mark
				continue;
			//std::vector<std::vector<std::pair<int, int> > > chars = findCharsBestMerge(ans.image);
			std::vector<std::vector<std::pair<int, int> > > chars = findCharsProjOnlyBestMerge(ans.image, 30, 25);
			//std::vector<std::vector<std::pair<int, int> > > chars = findChars(ans.image);
			for (int ord = 0; ord < ordcnt; ord++)
			{
				int mind = 5000000;
				int mindi = 0;
				//for (std::vector<std::pair<int, int> > c : chars)
				for (int i = 0; i < (int)chars.size(); i++)
				{
					Box<int> b = boundingBox(chars[i]);
					b.maxx += 1;
					b.maxy += 1;
					int dx = (b.minx + b.maxx) - 2 * charposx[ord];
					int dy = (b.miny + b.maxy) - 2 * charposy[ord];
					int td = dx * dx + dy * dy;
					if (mind > td)
					{
						mind = td;
						mindi = i;
					}
				}
				//printf("ord = %d, mindi = %d\n", ord, mindi);
				Box<int> b = boundingBox(chars[mindi]);
				b.maxx += 1;
				b.maxy += 1;
				charsmat.push_back(toMat(chars[mindi], b));
				//allchars.push_back(chars[mindi]);
				y.push_back(ord);
			}
		}
	}
	//write charsmat to a file
	for (cv::Mat &m : charsmat)
	{
		//normalize char
		m = normalizeCharNLN(m, 28, 28);
	}
	FILE *fp = fopen("data/chardata_chn1.2_automarked.bin", "wb");
	dumpMatVec(y, charsmat, fp);
	fclose(fp);
	for (int ord = 0; ; ord++)
	{
		std::vector<std::vector<float> > chars;
		for (int i = 0; i < (int)charsmat.size(); i++)
			if (y[i] == ord)
			{
				chars.push_back(toStdVector(charsmat[i]));
			}
		if (chars.size() == 0)
			break;
		displayChars(chars);
		cv::waitKey();
	}
	return 0;
}

int main()
{
	//return main_chars_mark();
	//return main_chars_pipe();
	//return main_char_pairs_encode();
	//return main_chars("data/chardata_chn1.2_normalized.bin");
	//return main_chars("data/chardata_his29_normalized.bin");
	//return main_chars_automark();
	//return main_chars_translate();
	//return main_chars_encode("data/encoded_features_chn1.2_normalized");
	//return main_chars_encode("data/encoded_features_his29_normalized");
	//
	//return main_kmeans("data/encoded_features_chn1.2", "data/encoded_features_chn1.2_1_300_clusters");
	//return main_kmeans("data/encoded_features_chn1.2_normalized", "data/encoded_features_chn1.2_normalized_50_clusters", 50);
	//return main_kmeans("data/encoded_features_his29_normalized", "data/encoded_features_his29_normalized_50_clusters", 50, 0.3);
	//
	//return main_kmeans_display("data/chardata_chn1.2.bin", "data/encoded_features_chn1.2_1_300_clusters");
	//return main_kmeans_display("data/chardata_chn1.2.bin", "data/encoded_features_chn1.2_normalized_50_clusters");
	//return main_kmeans_display_char_pair();
	//
	//return main_kmeans_feature("data/encoded_features_chn1.2_normalized", "data/chardata_chn1.2_normalized.bin", "data/encoded_features_chn1.2_normalized_50_clusters");
	//return main_kmeans_feature("data/encoded_features_chn1.2_normalized", "data/chardata_chn1.2_normalized.bin", "data/encoded_features_chn1.2_normalized_50_clusters");
	return main_kmeans_feature("data/encoded_features_his29_normalized", "data/chardata_his29_normalized.bin", "data/encoded_features_his29_normalized_50_clusters");
}
