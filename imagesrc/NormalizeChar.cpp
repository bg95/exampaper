#include <functional>
#include "NormalizeChar.h"

#define THR 0.5

void calL(float s[], float t[], int n)
{
	int l1[n], l2[n], l3[n], l4[n];
	//<0 invalid
	l1[0] = -n - 1;
	l3[0] = -n - 1;
	for (int i = 1; i < n; i++)
	{
		if (s[i - 1] > THR && s[i] <= THR) //black|white
			l1[i] = 0;
		else
			l1[i] = l1[i - 1] + 1;
		if (s[i - 1] <= THR && s[i] > THR) //white|black
			l3[i] = 0;
		else
			l3[i] = l3[i - 1] + 1;
	}
	l2[n - 1] = -n - 1;
	l4[n - 1] = -n - 1;
	for (int i = n - 2; i >= 0; i--)
	{
		if (s[i] > THR && s[i + 1] <= THR) //white|black
			l4[i] = 0;
		else
			l4[i] = l4[i + 1] + 1;
		if (s[i] <= THR && s[i + 1] > THR) //black|white
			l2[i] = 0;
		else
			l2[i] = l2[i + 1] + 1;
	}
	for (int i = 0; i < n; i++)
	{
		bool d1 = l1[i] >= 0;
		bool d2 = l2[i] >= 0;
		bool d3 = l3[i] >= 0;
		bool d4 = l4[i] >= 0;
		if (d1 && d2 && d3 && d4)
			t[i] = (l1[i] + l2[i] + l3[i] + l4[i]) / 2.0f;
		else if (!d1 && d4)
		{
			if (d3)
				t[i] = l4[i] + l3[i];
			else
				t[i] = 2 * n;
		}
		else if (d1 && !d4)
		{
			if (d2)
				t[i] = l2[i] + l1[i];
			else
				t[i] = 2 * n;
		}
		else if (d2 && d3)
			t[i] = 2 * n;
		else
			t[i] = 4 * n;
	//	printf("%d %d %d %d -> %f\n", l1[i], l2[i], l3[i], l4[i], t[i]);
	}
}

cv::Mat normalizeCharNLN(cv::Mat img, int maxx, int maxy)
{
	int r = img.rows, c = img.cols;
	//stage 1: calculate line density (inscribed circle)
	float s[std::max(r, c)], t[std::max(r, c)];
	float lx[r][c], ly[r][c];
	double rho[r][c], hx[r], hy[c];
	double sumrho;
	int samplex[maxx], sampley[maxy];
	for (int i = 0; i < r; i++)
	{
		for (int x = 0; x < c; x++)
			s[x] = img.at<float>(i, x);
		calL(s, t, c);
		for (int x = 0; x < c; x++)
			lx[i][x] = t[x] + 1;
	}
	for (int j = 0; j < c; j++)
	{
		for (int x = 0; x < r; x++)
			s[x] = img.at<float>(x, j);
		calL(s, t, r);
		for (int x = 0; x < r; x++)
			ly[x][j] = t[x] + 1;
	}
	/*
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			printf("lx[%d][%d] = %f\n", i, j, lx[i][j]);
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			printf("ly[%d][%d] = %f\n", i, j, ly[i][j]);
			*/
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
		{
			if (lx[i][j] + ly[i][j] < 6 * c)
				rho[i][j] = (double)c / std::min(lx[i][j], ly[i][j]);
			else
				rho[i][j] = 0;
		}
	/*
	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < c; j++)
			printf("%5.1lf", i, j, rho[i][j]);
		printf("\n");
	}
	*/
	//stage 2: calculate weight
	for (int i = 0; i < r; i++)
	{
		hx[i] = 0;
		for (int j = 0; j < c; j++)
			hx[i] += rho[i][j];
	}
	sumrho = 0;
	for (int i = 0; i < r; i++)
		sumrho += hx[i];
	for (int j = 0; j < c; j++)
	{
		hy[j] = 0;
		for (int i = 0; i < r; i++)
			hy[j] += rho[i][j];
	}
	//stage 3: resample
	int i = 0, j = 0;
	double ps = 0;
	for (int x = 0; x < maxx; x++)
	{
		while (ps * maxx < x * sumrho)
		{
			ps += hx[i];
			i++;
		}
		samplex[x] = i;
	}
	ps = 0;
	for (int y = 0; y < maxy; y++)
	{
		while (ps * maxx < y * sumrho)
		{
			ps += hy[j];
			j++;
		}
		sampley[y] = j;
	}
	cv::Mat res(maxx, maxy, img.type());
	for (int x = 0; x < maxx; x++)
		for (int y = 0; y < maxy; y++)
			res.at<float>(x, y) = 1.0 ? img.at<float>(samplex[x], sampley[y]) >= THR : 0.0;
	return res;
}
Matrix normalizeCharNLN(Matrix img, int maxx, int maxy)
{
	int r = img.rows, c = img.cols;
	//stage 1: calculate line density (inscribed circle)
	float s[std::max(r, c)], t[std::max(r, c)];
	float lx[r][c], ly[r][c];
	double rho[r][c], hx[r], hy[c];
	double sumrho;
	int samplex[maxx], sampley[maxy];
	for (int i = 0; i < r; i++)
	{
		for (int x = 0; x < c; x++)
			s[x] = img.at(i, x);
		calL(s, t, c);
		for (int x = 0; x < c; x++)
			lx[i][x] = t[x] + 1;
	}
	for (int j = 0; j < c; j++)
	{
		for (int x = 0; x < r; x++)
			s[x] = img.at(x, j);
		calL(s, t, r);
		for (int x = 0; x < r; x++)
			ly[x][j] = t[x] + 1;
	}
	/*
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			printf("lx[%d][%d] = %f\n", i, j, lx[i][j]);
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			printf("ly[%d][%d] = %f\n", i, j, ly[i][j]);
			*/
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
		{
			if (lx[i][j] + ly[i][j] < 6 * c)
				rho[i][j] = (double)c / std::min(lx[i][j], ly[i][j]);
			else
				rho[i][j] = 0;
		}
	/*
	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < c; j++)
			printf("%5.1lf", i, j, rho[i][j]);
		printf("\n");
	}
	*/
	//stage 2: calculate weight
	for (int i = 0; i < r; i++)
	{
		hx[i] = 0;
		for (int j = 0; j < c; j++)
			hx[i] += rho[i][j];
	}
	sumrho = 0;
	for (int i = 0; i < r; i++)
		sumrho += hx[i];
	for (int j = 0; j < c; j++)
	{
		hy[j] = 0;
		for (int i = 0; i < r; i++)
			hy[j] += rho[i][j];
	}
	//stage 3: resample
	int i = 0, j = 0;
	double ps = 0;
	for (int x = 0; x < maxx; x++)
	{
		while (ps * maxx < x * sumrho)
		{
			ps += hx[i];
			i++;
		}
		samplex[x] = i;
	}
	ps = 0;
	for (int y = 0; y < maxy; y++)
	{
		while (ps * maxx < y * sumrho)
		{
			ps += hy[j];
			j++;
		}
		sampley[y] = j;
	}
	Matrix res(maxx, maxy);
	for (int x = 0; x < maxx; x++)
		for (int y = 0; y < maxy; y++)
			res.at(x, y) = 1.0 ? img.at(samplex[x], sampley[y]) >= THR : 0.0;
	return res;
}
