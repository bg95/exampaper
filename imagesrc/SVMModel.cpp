#include <fstream>
#include <cstdlib>
#include <cmath>

#include "SVMModel.h"

SVMModel::SVMModel()
{
	loaded = false;
	model = 0;
	prob.l = -1;
	partial = false;
}

SVMModel::~SVMModel()
{
	if (model)
		svm_free_and_destroy_model(&model);
	if (prob.l != -1)
	{
		delete[] prob.y;
		for (int i = 0; i < prob.l; i++)
			delete[] prob.x[i];
		delete[] prob.x;
	}
}

void SVMModel::train(std::vector<std::vector<double> > features, std::vector<double> target)
{
	if (loaded)
	{
		fprintf(stderr, "No need to train if loaded\n");
		return;
	}
	//Normalization
	normalization.set(features);
	normalization.apply(features);
	int fn = features[0].size();
	prob.l = target.size();
	prob.y = new double[prob.l];
	for (int i = 0; i < prob.l; i++)
		prob.y[i] = target[i];
	prob.x = new svm_node *[prob.l];
	for (int i = 0; i < prob.l; i++)
	{
		prob.x[i] = new svm_node[fn + 1];
		toSVMNodes(features[i], prob.x[i]);
	}
	setDefaultParameters();
	fprintf(stderr, "SVM training...\n");
	const char *err = svm_check_parameter(&prob, &param);
	if (!err)
	{
		fprintf(stderr, "SVM error: %s\n", err);
		return;
	}

	model = svm_train(&prob, &param);
	loaded = true;
	fprintf(stderr, "SVM training finished\n");
}

double SVMModel::predict(std::vector<double> f)
{
	if (!model)
		return 0;
	normalization.apply(f);
	struct svm_node *x = new struct svm_node[f.size() + 1];
	toSVMNodes(f, x);
	double res = svm_predict(model, x);
	delete[] x;
	return res;
}

std::vector<double> SVMModel::predict(std::vector<std::vector<double> > f)
{
	std::vector<double> res;
	for (std::vector<double> &tf : f)
		res.push_back(predict(tf));
	return res;
}

std::vector<double> SVMModel::crossValidation(std::vector<std::vector<double> > features, std::vector<double> target, int nr_fold)
{
	if (loaded)
	{
		fprintf(stderr, "SVM loaded, will overwrite for cross validation\n");
	}
	//Normalization
	normalization.set(features);
	normalization.apply(features);
	int fn = features[0].size();
	prob.l = target.size();
	prob.y = new double[prob.l];
	for (int i = 0; i < prob.l; i++)
		prob.y[i] = target[i];
	prob.x = new svm_node *[prob.l];
	for (int i = 0; i < prob.l; i++)
	{
		prob.x[i] = new svm_node[fn + 1];
		toSVMNodes(features[i], prob.x[i]);
	}
	setDefaultParameters();
	fprintf(stderr, "SVM cross validation...\n");
	const char *err = svm_check_parameter(&prob, &param);
	std::vector<double> ret(prob.l);
	if (!err)
	{
		fprintf(stderr, "SVM error: %s\n", err);
		return ret;
	}

	svm_cross_validation(&prob, &param, nr_fold, ret.data());
	fprintf(stderr, "SVM cross validation finished\n");
	return ret;
}

void SVMModel::load(const char *name)
{
	std::string s(name);
	model = svm_load_model((s + ".svm").data());
	std::ifstream is((s + ".norm").data());
	normalization.load(is);
	if (model != 0)
		loaded = true;
}

void SVMModel::save(const char *name)
{
	std::string s(name);
	svm_save_model((s + ".svm").data(), model);
	std::ofstream os((s + ".norm").data());
	normalization.save(os);
}

void SVMModel::setPartialFeature(std::vector<int> ind)
{
	indices = ind;
	partial = true;
}

void SVMModel::setDefaultParameters()
{
	//param.svm_type = C_SVC;
	//param.svm_type = NU_SVR;
	param.svm_type = EPSILON_SVR;
	//param.svm_type = ONE_CLASS;
	param.kernel_type = RBF;
	//param.kernel_type = LINEAR;
	//param.kernel_type = POLY;
	param.degree = 3;
	param.gamma = 0.2;
	param.coef0 = 1;
	param.cache_size = 500;
	param.nu = 1;
	param.eps = 0.1;
	param.C = 10;
	param.nr_weight = 0;
}

void SVMModel::toSVMNodes(const std::vector<double> &f, struct svm_node *x)
{
	if (partial)
	{
		int j = 0;
		for (std::vector<int>::const_iterator iter = indices.begin(); iter != indices.end(); iter++)
		{
			x[j].index = j;
			x[j].value = f[*iter];
			j++;
		}
		x[j].index = -1;
	}
	else
	{
		for (int j = 0; j < f.size(); j++)
		{
			x[j].index = j;
			x[j].value = f[j];
		}
	}
	x[f.size()].index = -1;
}


void Normalization::set(const std::vector<std::vector<double> > &features)
{
	int m = features[0].size();
	avg.resize(m);
	sd.resize(m);
	for (int i = 0; i < m; i++)
	{
		avg[i] = 0;
		for (int j = 0; j < features.size(); j++)
			avg[i] += features[j][i];
		avg[i] /= features.size();
	}
	for (int i = 0; i < m; i++)
	{
		sd[i] = 0;
		for (int j = 0; j < features.size(); j++)
			sd[i] += (features[j][i] - avg[i]) * (features[j][i] - avg[i]);
		sd[i] = std::sqrt(sd[i] / features.size());
		sd[i] = std::max(sd[i], 1E-7); //avoid div by 0
	}
}

void Normalization::apply(std::vector<double> &f)
{
	for (int j = 0; j < avg.size(); j++)
		f[j] = (f[j] - avg[j]) / sd[j];
}

void Normalization::apply(std::vector<std::vector<double> > &features)
{
	for (int i = 0; i < features.size(); i++)
		apply(features[i]);
}

void Normalization::applyInverse(std::vector<double> &f)
{
	for (int j = 0; j < avg.size(); j++)
		f[j] = f[j] * sd[j] + avg[j];
}

void Normalization::applyInverse(std::vector<std::vector<double> > &features)
{
	for (int i = 0; i < features.size(); i++)
		applyInverse(features[i]);
}

void Normalization::load(std::istream &is)
{
	loadStdVector(avg, is);
	loadStdVector(sd, is);
}

void Normalization::save(std::ostream &os)
{
	saveStdVector(avg, os);
	saveStdVector(sd, os);
}
