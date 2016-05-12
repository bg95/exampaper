#pragma once

#include "svm.h"
#include <vector>

class Normalization
{
public:
	void set(const std::vector<std::vector<double> > &features);
	void apply(std::vector<double> &f);
	void apply(std::vector<std::vector<double> > &features);
	void applyInverse(std::vector<double> &f);
	void applyInverse(std::vector<std::vector<double> > &features);
	void load(std::istream &is);
	void save(std::ostream &os);
private:
	std::vector<double> avg, sd;
	template <class T> static void saveStdVector(std::vector<T> &v, std::ostream &os)
	{
		os << " " << v.size();
		for (int i = 0; i < v.size(); i++)
			os << " " << v[i];
		os << "\n";
	}
	template <class T> static void loadStdVector(std::vector<T> &v, std::istream &is)
	{
		int n;
		is >> n;
		v.resize(n);
		for (int i = 0; i < n; i++)
			is >> v[i];
	}
};

class SVMModel
{
public:
	SVMModel();
	~SVMModel();
	void train(std::vector<std::vector<double> > features, std::vector<double> target);
	double predict(std::vector<double> f);
	std::vector<double> predict(std::vector<std::vector<double> > f);
	std::vector<double> crossValidation(std::vector<std::vector<double> > features, std::vector<double> target, int nr_fold);
	void load(const char *name);
	void save(const char *name);
	bool isLoaded() const
	{
		return loaded;
	}
	void setPartialFeature(std::vector<int> ind);
	void setDefaultParameters();

private:
	struct svm_problem prob;
	struct svm_parameter param;
	struct svm_model *model;
	bool loaded;
	bool partial;
	std::vector<int> indices;
	void toSVMNodes(const std::vector<double> &f, struct svm_node *x);
	Normalization normalization;
};

