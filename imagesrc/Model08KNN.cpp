#include <cstdio>
#include <algorithm>
#include <fstream>

#include "Model08KNN.h"

Model08KNN::Model08KNN()
{
	setK(1);
	loaded = false;
}

void Model08KNN::train(std::vector<Matrix> samples, std::vector<int> type)
{
	if (loaded)
	{
		printf("No need to train if loaded\n");
		return;
	}
	this->type = type;
	features = extractFeatures(samples);
}

int Model08KNN::predict(Matrix s)
{
	std::vector<double> f = extractFeatures(s);
	std::vector<std::pair<double, int> > dist_and_type(features.size());
	if (partial_dist)
	{
		for (int i = 0; i < features.size(); i++)
		{
			dist_and_type[i] = std::pair<double, int>(squaredDistancePartial(f, features[i]), type[i]);
		}
	}
	else
	{
		for (int i = 0; i < features.size(); i++)
		{
			dist_and_type[i] = std::pair<double, int>(squaredDistance(f, features[i]), type[i]);
		}
	}
	//std::sort(dist_and_type.begin(), dist_and_type.end());
	kmin(dist_and_type, K);
	int cnt0 = 0;
	for (int i = 0; i < K && i < dist_and_type.size(); i++)
	{
		if (dist_and_type[i].second == 0)
			cnt0++;
	}
	if (cnt0 > K - cnt0)
		return 0;
	return 1;
}

void Model08KNN::kmin(std::vector<std::pair<double,int> > &a,int k,int l,int r)
{	
	if(r==-1)
		r=(int)a.size()-1;
	
	int i=l,j=r;
	int mid=(l+r)/2;
	
	for(;i<=j;)
	{
		for(;a[i]<a[mid];i++);
		if(i<=j){swap(a[i],a[mid]);mid=i++;}
		for(;a[j]>a[mid];j--);
		if(i<=j){swap(a[j],a[mid]);mid=j--;}
	}
	
	if(mid==k-1)
		return;
	else if(mid>k-1)
		Model08KNN::kmin(a,k,l,mid-1);
	else
		Model08KNN::kmin(a,k,mid+1,r);
}

void Model08KNN::load()
{
	Model08::load();

	std::ifstream is("features");
	if (!is)
		return;
	int fn;
	is >> fn;
	features.resize(fn);
	for (int i = 0; i < fn; i++)
		loadStdVector(features[i], is);
	loadStdVector(type, is);
	is >> K;
	loaded = true;
}

void Model08KNN::save()
{
	Model08::save();

	std::ofstream os("features");
	os << " " << features.size() << "\n";
	for (int i = 0; i < features.size(); i++)
		saveStdVector(features[i], os);
	os << "\n";
	saveStdVector(type, os);
	os << " " << K << "\n";
}

void Model08KNN::setK(int K)
{
	this->K = K;
}

void Model08KNN::setPartialDistance(std::vector<int> ind)
{
	indices = ind;
	partial_dist = true;
}

int Model08KNN::getNumFeatures() const
{
	if (features.size() == 0)
		return 0;
	return features[0].size();
}

double Model08KNN::squaredDistance(const std::vector<double> &a, const std::vector<double> &b)
{
	double s = 0.0;
	for (int i = 0; i < a.size(); i++)
		s += (a[i] - b[i]) * (a[i] - b[i]);
	return s;
}

double Model08KNN::squaredDistancePartial(const std::vector<double> &a, const std::vector<double> &b)
{
	double s = 0.0;
	for (std::vector<int>::const_iterator iter = indices.begin(); iter != indices.end(); iter++)
	{
		int i = *iter;
		s += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return s;
}
