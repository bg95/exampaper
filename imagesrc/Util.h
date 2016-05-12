#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

template<class T>
class Vector
{
public:
	Vector()
	{
	}
	Vector(const std::vector<T> &a)
	{
		setValue(a);
	}
	Vector(const cv::Mat &m)
	{
		v.clear();
		for (int i = 0; i < m.rows; i++)
			for (int j = 0; j < m.cols; j++)
				v.push_back(m.at<T>(i, j));
	}
	T &operator[](int s)
	{
		return v[s];
	}
	const T &operator[](int s) const
	{
		return v[s];
	}
	void setValue(std::vector<T> a)
	{
		v = a;
	}
	void setZeros()
	{
		for (int i = 0; i < v.size(); i++)
			v[i] = 0;
	}
	Vector<T> operator +(Vector<T> b) const
	{
		return Vector<T>(*this) += b;
	}
	Vector<T> operator +=(Vector<T> b)
	{
		for (int i = 0; i < v.size(); i++)
			v[i] += b.v[i];
		return *this;
	}
	Vector<T> operator -(Vector<T> b) const
	{
		return Vector<T>(*this) -= b;
	}
	Vector<T> operator -=(Vector<T> b)
	{
		for (int i = 0; i < v.size(); i++)
			v[i] -= b.v[i];
		return *this;
	}
	Vector<T> operator *(double k) const
	{
		return Vector<T>(*this) *= k;
	}
	Vector<T> operator *=(double k)
	{
		for (int i = 0; i < v.size(); i++)
			v[i] *= k;
		return *this;
	}
	friend Vector<T> operator *(double k, Vector<T> v);
	T abs() const
	{
		T s(0);
		for (int i = 0; i < v.size(); i++)
			s += v[i] * v[i];
		return s;
	}
	std::vector<T> v;
};

template<class T>
Vector<T> operator *(double k, Vector<T> v)
{
	return v * k;
}

template<class T>
T distance(Vector<T> a, Vector<T> b)
{
	return (a - b).abs();
}

template<class Matrix>
std::vector<Matrix> kMeans(std::vector<Matrix> samples, int clusters, std::vector<int> &cluster)
{
	if (samples.size() <= clusters)
		return samples;
	int n = samples.size();
	std::vector<Matrix> avrpos(clusters, samples[0]);
	std::vector<Matrix> cntr(clusters, samples[0]);
	int *clustersize = new int[clusters];
	//int *cluster = new int[n];
	cluster.resize(n);
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
			avrpos[j].setZeros();
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
			avrpos[mindiscntr] += samples[i];
			clustersize[mindiscntr]++;
		}
		printf("updates %d\n", updates);
		if (!flag)
			break;
		int k = 0;
		for (j = 0; j < clusters; j++)
		{
			if (clustersize[j] != 0)
			{
				avrpos[j] = avrpos[j] * (1.0 / (double)clustersize[j]);
				cntr[k++] = avrpos[j];
			}
		}
		clusters = k;
	}
	delete[] clustersize;
	//delete[] cluster;
	cntr.resize(clusters);
	return cntr;
}
