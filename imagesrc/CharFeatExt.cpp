#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#include "CharFeatExt.h"

const int CharFeatExt::PATCH_SIZE = 6;
const int CharFeatExt::CLUSTER_CNT = 40;
const cv::Size CharFeatExt::CHAR_SIZE(28, 28);

CharFeatExt::CharFeatExt()
{
	clusters_loaded = false;
}

void CharFeatExt::load(std::ifstream &is)
{
	//std::ifstream is("clusters");
	if (!is)
		return;
	int n;
	is >> n;
	clusters.clear();
	clusters.resize(n);
	for (int i = 0; i < n; i++)
		 loadFloatMat(clusters[i], is);
	clusters_loaded = true;
}

void CharFeatExt::save(std::ofstream &os)
{
	//std::ofstream os("clusters");
	os << " " << clusters.size() << "\n";
	for (int i = 0; i < clusters.size(); i++)
		saveFloatMat(clusters[i], os);
}

void CharFeatExt::init(std::vector<cv::Mat> samples)
{
	//resize characters to the same size
	for (cv::Mat &sample : samples)
		cv::resize(sample, sample, CHAR_SIZE);
	//Init feature extracting
	if (!clusters_loaded)
	{
		std::vector<cv::Mat> patches;
		for (int i = 0; i < samples.size(); i++)
			if (rand() % 10 == 0)
		{
			printf("Processing sample %d\n", i);
			for (int j = 0; j + PATCH_SIZE <= samples[i].rows; j++)
				for (int k = 0; k + PATCH_SIZE <= samples[i].cols; k++)
					patches.push_back(samples[i](cv::Range(j, j + PATCH_SIZE), cv::Range(k, k + PATCH_SIZE)));
		}
		printf("clustering...\n");
		clusters = kMeans(patches, CLUSTER_CNT);
		clusters_loaded = true;
	}
	else
	{
		fprintf(stderr, "Clusters already initialized\n");
	}
}

std::vector<double> CharFeatExt::extractPatchFeatures(cv::Mat sample)
{
	//std::vector<double> f;
	std::vector<double> sum(clusters.size(), 0);
	std::vector<double> d(clusters.size(), 0);
	double coe = 1.0 / (sample.rows * sample.cols);
	for (int j = 0; j + PATCH_SIZE <= sample.rows; j++)
		for (int k = 0; k + PATCH_SIZE <= sample.cols; k++)
		{
			for (int i = 0; i < clusters.size(); i++)
				d[i] = -distance(clusters[i], sample(cv::Range(j, j + PATCH_SIZE), cv::Range(k, k + PATCH_SIZE)));
			/*
			for (int i = 0; i < clusters.size(); i++)
				if (d[i] < 0)
					d[i] = 0;
					*/
			normalize(d);
			for (int i = 0; i < clusters.size(); i++)
				if (d[i] > 0)
					sum[i] += d[i] * coe;
		}
	normalize(sum);
	return sum;
	/*
	for (int i = 0; i < sum.size(); i++)
		f.push_back(sum[i]);
	return f;
	*/
}

std::vector<double> CharFeatExt::extractFeatures(cv::Mat sample)
{
	cv::Mat resized;
	cv::resize(sample, resized, CHAR_SIZE);
	std::vector<double> f;
	cv::Range l(0, CHAR_SIZE.width / 2), r(CHAR_SIZE.width / 2, CHAR_SIZE.width);
	cv::Range t(0, CHAR_SIZE.height / 2), b(CHAR_SIZE.height / 2, CHAR_SIZE.height);
	cv::Mat tl = resized(t, l), tr = resized(t, r), bl = resized(b, l), br = resized(b, r);
	append(f, extractPatchFeatures(tl));
	append(f, extractPatchFeatures(tr));
	append(f, extractPatchFeatures(bl));
	append(f, extractPatchFeatures(br));
	return f;
}

std::vector<double> CharFeatExt::extractSetFeatures(std::vector<cv::Mat> samples, int nb)
{
	nbuck = nb;
	int nf = clusters.size() * 4;
	std::vector<double> f(nf * nbuck, 0);  //need to be modified if # features changes
	for (int i = 0; i < samples.size(); i++)
	{
		std::vector<double> tf = extractFeatures(samples[i]);
		for (int j = 0; j < tf.size(); j++)
			f[(i % nbuck) * nf + j] += tf[j];
	}
	return f;
}

std::vector<std::string> CharFeatExt::getSetFeatureNames()
{
	std::vector<std::string> names;
	for (int b = 0; b < nbuck; b++)
	{
		for (int i = 0; i < clusters.size(); i++)
		{
			char str[256];
			sprintf(str, "chartl%d_b%d", i, b);
			names.push_back(str);
		}
		for (int i = 0; i < clusters.size(); i++)
		{
			char str[256];
			sprintf(str, "chartr%d_b%d", i, b);
			names.push_back(str);
		}
		for (int i = 0; i < clusters.size(); i++)
		{
			char str[256];
			sprintf(str, "charbl%d_b%d", i, b);
			names.push_back(str);
		}
		for (int i = 0; i < clusters.size(); i++)
		{
			char str[256];
			sprintf(str, "charbr%d_b%d", i, b);
			names.push_back(str);
		}
	}
	return names;
}

std::vector<cv::Mat> CharFeatExt::kMeans(std::vector<cv::Mat> samples, int clusters)
{
	if (samples.size() <= clusters)
		return samples;
	int n = samples.size();
	int sr = samples[0].rows, sc = samples[0].cols;
	std::vector<cv::Mat> avrpos(clusters, cv::Mat(sr, sc, CV_32FC1));
	std::vector<cv::Mat> cntr(clusters, cv::Mat(sr, sc, CV_32FC1));
	int *clustersize = new int[clusters];
	int *cluster = new int[n];
	int i, j;
	for (i = 0; i < n; i++)
		cluster[i] = -1;
	for (j = 0; j < clusters; j++)
	{
		int k;
		do
			k = rand() % n;
		while (cluster[k] != -1);
		cluster[k] = j;
		cntr[j] = samples[k];
	}
	bool flag;
	int iter = 0;
	while (1)
	{
		printf("kMeans iteration #%d\n", iter);
		iter++;
		
		flag = false;
		int updates = 0;
		for (j = 0; j < clusters; j++)
			avrpos[j] = cv::Mat(cv::Mat::zeros(sr, sc, CV_32FC1));
		for (j = 0; j < clusters; j++)
			clustersize[j] = 0;
		for (i = 0; i < n; i++)
		{
			double mindis, tdis;
			int mindiscntr;
			mindis = 1.0 / 0.0; //Infinity
			for (j = 0; j < clusters; j++)
			{
				tdis = distance(samples[i], cntr[j]);
				if (tdis < mindis)
				{
					mindis = tdis;
					mindiscntr = j;
				}
			}
			if (cluster[i] != mindiscntr)
			{
				flag = true;
				updates++;
				cluster[i] = mindiscntr;
			}
			avrpos[mindiscntr] = avrpos[mindiscntr] + samples[i];
			clustersize[mindiscntr]++;
			/*
			fprintf(stderr, "mindiscntr of %d is %d\n", i, mindiscntr);
			{
				cv::Mat c = samples[i];
				cv::Mat mag;
				cv::resize(c, mag, cv::Size(240, 240), cv::INTER_NEAREST);
				char str[256];
				sprintf(str, "sample");
				cv::imshow(str, mag);
			}
			for (int k = 0; k < avrpos.size(); k++)
			{
				cv::Mat c = avrpos[k] / clustersize[k] / 2;
				cv::Mat mag;
				cv::resize(c, mag, cv::Size(240, 240), cv::INTER_NEAREST);
				char str[256];
				sprintf(str, "avgpos%d", k);
				cv::imshow(str, mag);
			}
				cv::waitKey();
				*/
		}
		printf("updates %d\n", updates);
		if (!flag)
			break;
		int k = 0;
		for (j = 0; j < clusters; j++)
		{
			if (clustersize[j] != 0)
			{
				avrpos[j] = avrpos[j] / (double)clustersize[j];
				cntr[k++] = avrpos[j];
			}
		}
		clusters = k;
		/*
		for (cv::Mat &c : cntr)
		{
			cv::Mat mag;
			cv::resize(c, mag, cv::Size(240, 240), cv::INTER_NEAREST);
			cv::imshow("center", mag);
			cv::waitKey();
		}
		*/
	}
	delete[] clustersize;
	delete[] cluster;
	cntr.resize(clusters, cv::Mat(sr, sc, CV_32FC1));
	return cntr;
}

double CharFeatExt::distance(cv::Mat a, cv::Mat b)
{
	//return cv::norm(a - b);
	double s = 0;
	for (int i = 0; i < a.rows; i++)
		for (int j = 0; j < a.cols; j++)
			s += (a.at<float>(i, j) - b.at<float>(i, j)) * (a.at<float>(i, j) - b.at<float>(i, j));
	return s;
}

void CharFeatExt::normalize(std::vector<double> &v)
{
	double a = 0.0;
	for (int i = 0; i < v.size(); i++)
		a += v[i];
	a /= v.size();
	for (int i = 0; i < v.size(); i++)
		v[i] -= a;
	double s = 0.0;
	for (int i = 0; i < v.size(); i++)
		s += v[i] * v[i];
	s /= v.size();
	s = std::sqrt(s);
	for (int i = 0; i < v.size(); i++)
		v[i] /= s;
}

void CharFeatExt::loadFloatMat(cv::Mat &m, std::istream &is)
{
	int r, c;
	is >> r >> c;
	m = cv::Mat::zeros(r, c, CV_32FC1);
	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++)
			is >> m.at<float>(i, j);
}

void CharFeatExt::saveFloatMat(const cv::Mat &m, std::ostream &os)
{
	os << " " << m.rows << " " << m.cols;
	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++)
			os << " " << m.at<float>(i, j);
	os << "\n";
}
