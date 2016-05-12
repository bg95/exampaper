#ifndef MODEL08KNN_H
#define MODEL08KNN_H

#include "Model08.h"

class Model08KNN : public Model08
{
public:
	Model08KNN();
	void train(std::vector<Matrix> samples, std::vector<int> type);
	int predict(Matrix s);
	int predict(std::vector<double> features);
	void load();
	void save();
	void setK(int K);
	void setPartialDistance(std::vector<int> ind);
	int getNumFeatures() const;

private:
	double squaredDistance(const std::vector<double> &a, const std::vector<double> &b);
	double squaredDistancePartial(const std::vector<double> &a, const std::vector<double> &b);
	void kmin(std::vector<std::pair<double,int> > &a,int k,int l=0,int r=-1);
	std::vector<std::vector<double> > features;
	std::vector<int> type;
	int K;
	bool loaded, partial_dist;
	std::vector<int> indices;
};

#endif
