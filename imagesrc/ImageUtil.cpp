#include "ImageUtil.h"

void showBoxes(const cv::Mat &img, const std::vector<Box<int> > &boundingboxes, const char *name)
{
	/*
	cv::Mat rectimg;
	img.convertTo(rectimg, CV_32FC3);
	for (Box<int> b : boundingboxes)
		cv::rectangle(rectimg, b.toRect(), cv::Scalar(1, 0, 0));
	cv::imshow(name, rectimg);
	cv::waitKey();
	*/
}

std::vector<std::vector<std::pair<int, int> > > findChars(cv::Mat img)
{
	const int sizeth = 10; //discard components that are smaller than this size
	//const int sizethu = 200; //discard components that are larger than this size (this may be the black border)
	const double blackth = 0.05, blackthu = 0.9; //discard components that are too white (frame), or too black (black markers)
	const double hspth = 2.3, vspth = 2;
	const double hspthl = 0.9, vspthl = 0.7;
	const int splititer = 1;
	const int spadj = 2;
	const double interth = 0.5, smallth = 0.8;
	const int splitsegcost = 2;
	std::vector<std::vector<std::pair<int, int> > > components;
	std::vector<Box<int> > boundingboxes;
	allComponents(img, components, boundingboxes, sizeth, blackth, blackthu);
	double tsx;
	double tsy;
	std::vector<int> xsize, ysize;
	cv::Mat imgcpy = img;
	for (int si = 0; si < splititer; si++)
	{
		img = imgcpy;
		//pick a typical character size
		xsize.clear();
		ysize.clear();
		for (Box<int> b : boundingboxes)
		{
			if (b.rangeX() >= sizeth)
				xsize.push_back(b.rangeX());
			if (b.rangeY() >= sizeth)
				ysize.push_back(b.rangeY());
		}
		std::sort(xsize.begin(), xsize.end());
		std::sort(ysize.begin(), ysize.end());
		tsx = xsize[xsize.size() / 2];
		tsy = ysize[ysize.size() / 2];
		//fprintf(stderr, "typ size %lf,%lf\n", tsx, tsy);
		//split horizontally
		for (int i = 0; i < components.size(); i++)
			if (boundingboxes[i].rangeX() >= hspth * tsx)
			{
				std::vector<std::vector<int> > hp(boundingboxes[i].rangeX());
				for (std::pair<int, int> p : components[i])
					hp[p.first - boundingboxes[i].minx].push_back(p.second);
				std::vector<double> hpc(hp.size());
				for (int i = 0; i < hp.size(); i++)
				{
					std::sort(hp[i].begin(), hp[i].end());
					hpc[i] = hp[i].size();
					for (int j = 1; j < hp[i].size(); j++)
						if (hp[i][j - 1] != hp[i][j] - 1)
							hpc[i] += splitsegcost;
				}
				std::vector<int> hs = bestSplit(hpc, hspthl * tsx, hspth * tsx);
				for (int s : hs)
					cv::line(img, cv::Point(boundingboxes[i].miny, s + boundingboxes[i].minx), cv::Point(boundingboxes[i].maxy, s + boundingboxes[i].minx), cv::Scalar(0, 0, 0));
			}
		//recompute components
		allComponents(img, components, boundingboxes, sizeth, blackth, blackthu);
		//split vertically
		for (int i = 0; i < components.size(); i++)
			if (boundingboxes[i].rangeY() >= vspth * tsy)
			{
				std::vector<std::vector<int> > vp(boundingboxes[i].rangeY());
				for (std::pair<int, int> p : components[i])
					vp[p.second - boundingboxes[i].miny].push_back(p.first);
				std::vector<double> vpc(vp.size());
				for (int i = 0; i < vp.size(); i++)
				{
					std::sort(vp[i].begin(), vp[i].end());
					vpc[i] = vp[i].size();
					for (int j = 1; j < vp[i].size(); j++)
						if (vp[i][j - 1] != vp[i][j] - 1)
							vpc[i] += splitsegcost;
				}
				std::vector<int> vs = bestSplit(vpc, vspthl * tsy, vspth * tsy);
				for (int s : vs)
					cv::line(img, cv::Point(s + boundingboxes[i].miny, boundingboxes[i].minx), cv::Point(s + boundingboxes[i].miny, boundingboxes[i].maxx), cv::Scalar(0, 0, 0));
			}
		//recompute components
		allComponents(img, components, boundingboxes, sizeth, blackth, blackthu);
	}
	//merge small components
	std::vector<bool> deleted;
	bool flag = true;
	while (flag)
	{
		deleted.resize(components.size());
		for (int i = 0; i < deleted.size(); i++)
			deleted[i] = false;
		flag = false;
		for (int i = 0; i < components.size(); i++)
			if (!deleted[i])
				for (int j = i + 1; j < components.size(); j++)
					if (!deleted[j])
					{
						Box<int> inter = intersect(boundingboxes[i], boundingboxes[j]);
						if (inter.area() > boundingboxes[i].area() * interth ||
								inter.area() > boundingboxes[j].area() * interth)
						{
							//merge j into i
							deleted[j] = true;
							flag = true;
							for (auto x : components[j])
								components[i].push_back(x);
							boundingboxes[i] = uni(boundingboxes[i], boundingboxes[j]);
						}
					}
		for (int i = 0; i < components.size(); i++)
			if (!deleted[i] &&
					(boundingboxes[i].rangeX() < smallth * tsx ||
					boundingboxes[i].rangeY() < smallth * tsy))
				for (int j = i + 1; j < components.size(); j++)
					if (!deleted[j] && 
							(boundingboxes[j].rangeX() < smallth * tsx ||
							boundingboxes[j].rangeY() < smallth * tsy))
					{
						Box<int> u = uni(boundingboxes[i], boundingboxes[j]);
						if (u.rangeX() < 2 * tsx && u.rangeY() < 2 * tsy &&
								dist2<double>(tsx, tsy, boundingboxes[i].rangeX(), boundingboxes[i].rangeY()) +
								dist2<double>(tsx, tsy, boundingboxes[j].rangeX(), boundingboxes[j].rangeY()) >
								dist2<double>(tsx, tsy, u.rangeX(), u.rangeY()))
						{
							//merge j into i
							deleted[j] = true;
							flag = true;
							for (auto x : components[j])
								components[i].push_back(x);
							boundingboxes[i] = u;
						}
					}
		int m = 0;
		for (int i = 0; i < components.size(); i++)
			if (!deleted[i])
			{
				components[m].swap(components[i]);
				boundingboxes[m] = boundingboxes[i];
				m++;
			}
		components.resize(m);
		boundingboxes.resize(m);
	}
	//delete small components that are not merged
	int m = 0;
	for (int i = 0; i < components.size(); i++)
		if (boundingboxes[i].rangeX() >= tsx * smallth && boundingboxes[i].rangeY() >= tsy * smallth)
		{
			components[m].swap(components[i]);
			boundingboxes[m] = boundingboxes[i];
			m++;
		}
	components.resize(m);
	boundingboxes.resize(m);
	/*
	for (int i = 0; i < components.size(); i++)
	{
		fprintf(stderr, "size %d,%d blackness %lf\n", boundingboxes[i].rangeX(), boundingboxes[i].rangeY(), components[i].size() / (double)boundingboxes[i].area());
	}
	cv::Mat rectimg;
	img.convertTo(rectimg, CV_32FC3);
	for (Box<int> b : boundingboxes)
		cv::rectangle(rectimg, b.toRect(), cv::Scalar(1, 0, 0));
	cv::imshow("rectimg", rectimg);
	cv::waitKey();
	*/
	return components;
}
std::vector<std::vector<std::pair<int, int> > > findCharsAnother(cv::Mat img, double tsx, double tsy) //uses only projection
{
	const int sizeth = 10; //discard components that are smaller than this size
	//const int sizethu = 200; //discard components that are larger than this size (this may be the black border)
	const double blackth = 0.05, blackthu = 0.9; //discard components that are too white (frame), or too black (black markers)
	const double hspth = 2.3, vspth = 2;
	const double hspthl = 0.9, vspthl = 0.7;
	const int splititer = 2;
	const int spadj = 2;
	const double interth = 0.5, smallth = 0.8;
	const int splitsegcost = 2, blankcost = 1;
	std::vector<std::vector<std::pair<int, int> > > components;
	std::vector<Box<int> > boundingboxes;
	std::vector<int> xsize, ysize;
	//fprintf(stderr, "typ size %lf,%lf\n", tsx, tsy);
	//split horizontally
	Box<int> box(0, img.rows, 0, img.cols);
	std::vector<std::vector<int> > hp(box.rangeX());
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (img.at<float>(i, j) >= 0.9) //is black originally
				hp[i - box.minx].push_back(j);
	std::vector<double> hpc(hp.size());
	for (int i = 0; i < hp.size(); i++)
	{
		std::sort(hp[i].begin(), hp[i].end());
		hpc[i] = hp[i].size() + blankcost;
		for (int j = 1; j < hp[i].size(); j++)
			if (hp[i][j - 1] != hp[i][j] - 1)
				hpc[i] += splitsegcost;
	}
	std::vector<int> hs = bestSplit(hpc, hspthl * tsx, hspth * tsx);
	for (int s : hs)
		cv::line(img, cv::Point(box.miny, s + box.minx), cv::Point(box.maxy, s + box.minx), cv::Scalar(0, 0, 0));
	//split vertically
	std::sort(hs.begin(), hs.end());
	for (int k = 0; k <= hs.size(); k++)
	{
		cv::Range rangex(k ? hs[k - 1] : 0, k < hs.size() ? hs[k] : img.rows);
		int offsetx = rangex.start;
		cv::Mat timg = img(rangex, cv::Range::all());
		Box<int> tbox(0, timg.rows, 0, timg.cols);
		std::vector<std::vector<int> > vp(tbox.rangeY());
		for (int i = 0; i < timg.rows; i++)
			for (int j = 0; j < timg.cols; j++)
				if (timg.at<float>(i, j) >= 0.9) //is black originally
					vp[j - tbox.miny].push_back(i);
		std::vector<double> vpc(vp.size());
		for (int i = 0; i < vp.size(); i++)
		{
			std::sort(vp[i].begin(), vp[i].end());
			vpc[i] = vp[i].size() + blankcost;
			for (int j = 1; j < vp[i].size(); j++)
				if (vp[i][j - 1] != vp[i][j] - 1)
					vpc[i] += splitsegcost;
		}
		std::vector<int> vs = bestSplit(vpc, vspthl * tsy, vspth * tsy);
		for (int s : vs)
			cv::line(img, cv::Point(s + tbox.miny, tbox.minx + offsetx), cv::Point(s + tbox.miny, tbox.maxx + offsetx), cv::Scalar(0, 0, 0));
		std::sort(vs.begin(), vs.end());
		for (int s = 0; s <= vs.size(); s++)
		{
			components.push_back(std::vector<std::pair<int, int> >());
			cv::Range rangey(s ? vs[s - 1] : 0, s < vs.size() ? vs[s] : timg.cols);
			int offsety = rangey.start;
			cv::Mat ttimg = timg(cv::Range::all(), rangey);
			for (int ti = 0; ti < ttimg.rows; ti++)
				for (int tj = 0; tj < ttimg.cols; tj++)
					if (ttimg.at<float>(ti, tj) >= 0.9) //is black originally
						components.back().push_back(std::make_pair(ti + offsetx, tj + offsety));
			if (components.back().empty())
				components.pop_back();
		}
	}
	/*
	for (int i = 0; i < components.size(); i++)
	{
		fprintf(stderr, "size %d,%d blackness %lf\n", boundingboxes[i].rangeX(), boundingboxes[i].rangeY(), components[i].size() / (double)boundingboxes[i].area());
	}
	*/
	//delete too small components
	for (int i = 0; i < components.size(); i++)
		boundingboxes.push_back(boundingBox(components[i]));
	int m = 0;
	for (int i = 0; i < components.size(); i++)
		if (boundingboxes[i].rangeX() >= tsx * smallth && boundingboxes[i].rangeY() >= tsy * smallth)
		{
			components[m].swap(components[i]);
			boundingboxes[m] = boundingboxes[i];
			m++;
		}
	components.resize(m);
	boundingboxes.resize(m);
	//show result
	/*
	cv::Mat rectimg;
	img.convertTo(rectimg, CV_32FC3);
	for (Box<int> b : boundingboxes)
		cv::rectangle(rectimg, b.toRect(), cv::Scalar(1, 0, 0));
	cv::imshow("rectimg", rectimg);
	cv::waitKey();
	*/
	return components;
}
std::vector<std::vector<std::pair<int, int> > > findCharsProjOnlyBestMerge(cv::Mat img, double tsx, double tsy) //uses only projection
{
	const int sizeth = 10; //discard components that are smaller than this size
	//const int sizethu = 200; //discard components that are larger than this size (this may be the black border)
	const double blackth = 0.05, blackthu = 0.9; //discard components that are too white (frame), or too black (black markers)
	const double hspth = 2.3, vspth = 2;
	const double hspthl = 0.9, vspthl = 0.7;
	const int splititer = 2;
	const int spadj = 2;
	const double interth = 0.5, smallth = 0.8;
	const int splitsegcost = 2, blankcost = 1;
	std::vector<std::vector<std::pair<int, int> > > components;
	std::vector<Box<int> > boundingboxes;
	std::vector<int> xsize, ysize;
	//fprintf(stderr, "typ size %lf,%lf\n", tsx, tsy);
	//split horizontally
	Box<int> box(0, img.rows, 0, img.cols);
	std::vector<std::vector<int> > hp(box.rangeX());
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (img.at<float>(i, j) >= 0.9) //is black originally
				hp[i - box.minx].push_back(j);
	std::vector<double> hpc(hp.size());
	for (int i = 0; i < hp.size(); i++)
	{
		std::sort(hp[i].begin(), hp[i].end());
		hpc[i] = hp[i].size() + blankcost;
		for (int j = 1; j < hp[i].size(); j++)
			if (hp[i][j - 1] != hp[i][j] - 1)
				hpc[i] += splitsegcost;
	}
	std::vector<int> hs = bestSplit(hpc, hspthl * tsx, hspth * tsx);
	for (int s : hs)
		cv::line(img, cv::Point(box.miny, s + box.minx), cv::Point(box.maxy, s + box.minx), cv::Scalar(0, 0, 0));
	//split vertically
	std::sort(hs.begin(), hs.end());
	for (int k = 0; k <= hs.size(); k++)
	{
		cv::Range rangex(k ? hs[k - 1] : 0, k < hs.size() ? hs[k] : img.rows);
		int offsetx = rangex.start;
		cv::Mat timg = img(rangex, cv::Range::all());
		Box<int> tbox(0, timg.rows, 0, timg.cols);
		std::vector<std::vector<int> > vp(tbox.rangeY());
		for (int i = 0; i < timg.rows; i++)
			for (int j = 0; j < timg.cols; j++)
				if (timg.at<float>(i, j) >= 0.9) //is black originally
					vp[j - tbox.miny].push_back(i);
		std::vector<double> vpc(vp.size());
		for (int i = 0; i < vp.size(); i++)
		{
			std::sort(vp[i].begin(), vp[i].end());
			vpc[i] = vp[i].size() + blankcost;
			for (int j = 1; j < vp[i].size(); j++)
				if (vp[i][j - 1] != vp[i][j] - 1)
					vpc[i] += splitsegcost;
		}
		std::vector<int> vs = bestSplit(vpc, vspthl * tsy, vspth * tsy);
		for (int s : vs)
			cv::line(img, cv::Point(s + tbox.miny, tbox.minx + offsetx), cv::Point(s + tbox.miny, tbox.maxx + offsetx), cv::Scalar(0, 0, 0));
		std::sort(vs.begin(), vs.end());
		for (int s = 0; s <= vs.size(); s++)
		{
			components.push_back(std::vector<std::pair<int, int> >());
			cv::Range rangey(s ? vs[s - 1] : 0, s < vs.size() ? vs[s] : timg.cols);
			int offsety = rangey.start;
			cv::Mat ttimg = timg(cv::Range::all(), rangey);
			for (int ti = 0; ti < ttimg.rows; ti++)
				for (int tj = 0; tj < ttimg.cols; tj++)
					if (ttimg.at<float>(ti, tj) >= 0.9) //is black originally
						components.back().push_back(std::make_pair(ti + offsetx, tj + offsety));
			if (components.back().empty())
				components.pop_back();
		}
	}
	/*
	for (int i = 0; i < components.size(); i++)
	{
		fprintf(stderr, "size %d,%d blackness %lf\n", boundingboxes[i].rangeX(), boundingboxes[i].rangeY(), components[i].size() / (double)boundingboxes[i].area());
	}
	*/
	boundingboxes.resize(components.size());
	for (int i = 0; i < components.size(); i++)
		boundingboxes[i] = boundingBox(components[i]);
	//merge overlapping components
	std::vector<bool> deleted;
	int m = 0;
	deleted.resize(components.size());
	for (int i = 0; i < deleted.size(); i++)
		deleted[i] = false;
	for (int i = 0; i < components.size(); i++)
		if (!deleted[i])
			for (int j = i + 1; j < components.size(); j++)
				if (!deleted[j])
				{
					Box<int> inter = intersect(boundingboxes[i], boundingboxes[j]);
					if (inter.area() > boundingboxes[i].area() * interth ||
							inter.area() > boundingboxes[j].area() * interth)
					{
						//merge j into i
						deleted[j] = true;
						for (auto x : components[j])
							components[i].push_back(x);
						boundingboxes[i] = uni(boundingboxes[i], boundingboxes[j]);
					}
				}
	m = 0;
	for (int i = 0; i < components.size(); i++)
		if (!deleted[i])
		{
			components[m].swap(components[i]);
			boundingboxes[m] = boundingboxes[i];
			m++;
		}
	components.resize(m);
	boundingboxes.resize(m);

	//best merging
	deleted.resize(components.size());
	for (int i = 0; i < deleted.size(); i++)
		deleted[i] = false;
	std::vector<std::pair<int, int> > merge;
	std::vector<int> rank;
	merge = bestMerge(img, components, boundingboxes, tsx, tsy, blankcost, splitsegcost, hspthl, hspth, rank);
	//reorder components
	for (int i = 0; i < rank.size(); i++)
		while (rank[i] != i)
		{
			components[i].swap(components[rank[i]]);
			std::swap(rank[i], rank[rank[i]]);
		}
	for (std::pair<int, int> &tm : merge)
	{
		tm.first = rank[tm.first];
		tm.second = rank[tm.second];
	}
	for (std::pair<int, int> tm : merge)
	{
		int i = tm.first, j = tm.second;
		if (i != j)
		{
			//merge j into i
			deleted[j] = true;
			for (auto x : components[j])
				components[i].push_back(x);
			boundingboxes[i] = uni(boundingboxes[i], boundingboxes[j]);
		}
	}
	m = 0;
	for (int i = 0; i < components.size(); i++)
		if (!deleted[i])
		{
			components[m].swap(components[i]);
			boundingboxes[m] = boundingboxes[i];
			m++;
		}
	components.resize(m);
	boundingboxes.resize(m);

	showBoxes(img, boundingboxes, "merge"); //////////////////////////////////////

	//delete small components that are not merged
	//delete components that are too black or white
	m = 0;
	for (int i = 0; i < components.size(); i++)
		if (boundingboxes[i].rangeX() >= tsx * smallth && boundingboxes[i].rangeY() >= tsy * smallth)
		{
			double b = components[i].size() / (double)boundingboxes[i].area();
			if (b < blackth || b > blackthu)
				continue;
			components[m].swap(components[i]);
			boundingboxes[m] = boundingboxes[i];
			m++;
		}
	components.resize(m);
	boundingboxes.resize(m);
	showBoxes(img, boundingboxes, "delete"); //////////////////////////////////////

	return components;
}
std::vector<std::vector<std::pair<int, int> > > findCharsBestMerge(cv::Mat img)
{
	const int sizeth = 10; //discard components that are smaller than this size
	//const int sizethu = 200; //discard components that are larger than this size (this may be the black border)
	const double blackth = 0.05, blackthu = 0.9; //discard components that are too white (frame), or too black (black markers)
	const double hspth = 2.3, vspth = 2;
	const double hspthl = 0.9, vspthl = 0.7;
	const int splititer = 1;
	const int spadj = 2;
	const double interth = 0.5, smallth = 0.8;
	const int splitsegcost = 2, blankcost = 1;
	std::vector<std::vector<std::pair<int, int> > > components;
	std::vector<Box<int> > boundingboxes;
	allComponents(img, components, boundingboxes, sizeth, blackth, blackthu);
	double tsx;
	double tsy;
	std::vector<int> xsize, ysize;
	cv::Mat imgcpy = img;
	showBoxes(img, boundingboxes, "components"); //////////////////////////////////////
	for (int si = 0; si < splititer; si++)
	{
		img = imgcpy;
		//pick a typical character size
		xsize.clear();
		ysize.clear();
		for (Box<int> b : boundingboxes)
		{
			if (b.rangeX() >= sizeth)
				xsize.push_back(b.rangeX());
			if (b.rangeY() >= sizeth)
				ysize.push_back(b.rangeY());
		}
		std::sort(xsize.begin(), xsize.end());
		std::sort(ysize.begin(), ysize.end());
		tsx = xsize[xsize.size() / 2];
		tsy = ysize[ysize.size() / 2];
		//fprintf(stderr, "typ size %lf,%lf\n", tsx, tsy);
		//split horizontally
		for (int i = 0; i < components.size(); i++)
			if (boundingboxes[i].rangeX() >= hspth * tsx)
			{
				std::vector<std::vector<int> > hp(boundingboxes[i].rangeX());
				for (std::pair<int, int> p : components[i])
					hp[p.first - boundingboxes[i].minx].push_back(p.second);
				std::vector<double> hpc(hp.size());
				for (int i = 0; i < hp.size(); i++)
				{
					std::sort(hp[i].begin(), hp[i].end());
					hpc[i] = hp[i].size() + blankcost;
					for (int j = 1; j < hp[i].size(); j++)
						if (hp[i][j - 1] != hp[i][j] - 1)
							hpc[i] += splitsegcost;
				}
				std::vector<int> hs = bestSplit(hpc, hspthl * tsx, hspth * tsx);
				for (int s : hs)
					cv::line(img, cv::Point(boundingboxes[i].miny, s + boundingboxes[i].minx), cv::Point(boundingboxes[i].maxy, s + boundingboxes[i].minx), cv::Scalar(0, 0, 0));
			}
		//recompute components
		allComponents(img, components, boundingboxes, sizeth, blackth, blackthu);
		//split vertically
		for (int i = 0; i < components.size(); i++)
			if (boundingboxes[i].rangeY() >= vspth * tsy)
			{
				std::vector<std::vector<int> > vp(boundingboxes[i].rangeY());
				for (std::pair<int, int> p : components[i])
					vp[p.second - boundingboxes[i].miny].push_back(p.first);
				std::vector<double> vpc(vp.size());
				for (int i = 0; i < vp.size(); i++)
				{
					std::sort(vp[i].begin(), vp[i].end());
					vpc[i] = vp[i].size() + blankcost;
					for (int j = 1; j < vp[i].size(); j++)
						if (vp[i][j - 1] != vp[i][j] - 1)
							vpc[i] += splitsegcost;
				}
				std::vector<int> vs = bestSplit(vpc, vspthl * tsy, vspth * tsy);
				for (int s : vs)
					cv::line(img, cv::Point(s + boundingboxes[i].miny, boundingboxes[i].minx), cv::Point(s + boundingboxes[i].miny, boundingboxes[i].maxx), cv::Scalar(0, 0, 0));
			}
		//recompute components
		allComponents(img, components, boundingboxes, sizeth, blackth, blackthu);
	}
	showBoxes(img, boundingboxes, "split"); //////////////////////////////////////
	//merge overlapping components
	std::vector<bool> deleted;
	int m = 0;
	deleted.resize(components.size());
	for (int i = 0; i < deleted.size(); i++)
		deleted[i] = false;
	for (int i = 0; i < components.size(); i++)
		if (!deleted[i])
			for (int j = i + 1; j < components.size(); j++)
				if (!deleted[j])
				{
					Box<int> inter = intersect(boundingboxes[i], boundingboxes[j]);
					if (inter.area() > boundingboxes[i].area() * interth ||
							inter.area() > boundingboxes[j].area() * interth)
					{
						//merge j into i
						deleted[j] = true;
						for (auto x : components[j])
							components[i].push_back(x);
						boundingboxes[i] = uni(boundingboxes[i], boundingboxes[j]);
					}
				}
	m = 0;
	for (int i = 0; i < components.size(); i++)
		if (!deleted[i])
		{
			components[m].swap(components[i]);
			boundingboxes[m] = boundingboxes[i];
			m++;
		}
	components.resize(m);
	boundingboxes.resize(m);

	//best merging
	deleted.resize(components.size());
	for (int i = 0; i < deleted.size(); i++)
		deleted[i] = false;
	std::vector<std::pair<int, int> > merge;
	std::vector<int> rank;
	merge = bestMerge(img, components, boundingboxes, tsx, tsy, blankcost, splitsegcost, hspthl, hspth, rank);
	//reorder components
	for (int i = 0; i < rank.size(); i++)
		while (rank[i] != i)
		{
			components[i].swap(components[rank[i]]);
			std::swap(rank[i], rank[rank[i]]);
		}
	for (std::pair<int, int> &tm : merge)
	{
		tm.first = rank[tm.first];
		tm.second = rank[tm.second];
	}
	for (std::pair<int, int> tm : merge)
	{
		int i = tm.first, j = tm.second;
		if (i != j)
		{
			//merge j into i
			deleted[j] = true;
			for (auto x : components[j])
				components[i].push_back(x);
			boundingboxes[i] = uni(boundingboxes[i], boundingboxes[j]);
		}
	}
	m = 0;
	for (int i = 0; i < components.size(); i++)
		if (!deleted[i])
		{
			components[m].swap(components[i]);
			boundingboxes[m] = boundingboxes[i];
			m++;
		}
	components.resize(m);
	boundingboxes.resize(m);

	showBoxes(img, boundingboxes, "merge"); //////////////////////////////////////

	//delete small components that are not merged
	m = 0;
	for (int i = 0; i < components.size(); i++)
		if (boundingboxes[i].rangeX() >= tsx * smallth && boundingboxes[i].rangeY() >= tsy * smallth)
		{
			components[m].swap(components[i]);
			boundingboxes[m] = boundingboxes[i];
			m++;
		}
	components.resize(m);
	boundingboxes.resize(m);
	showBoxes(img, boundingboxes, "delete"); //////////////////////////////////////

	return components;
}

std::vector<std::pair<int, int> > floodfill(const cv::Mat &img, int i, int j, cv::Mat &visit, int stamp)
{
	const int dx[] = {0, 1, 0, -1}, dy[] = {1, 0, -1, 0};
	const float thr = 0.3;
	float orig = img.at<float>(i, j);
	std::vector<std::pair<int, int> > stack, ret;
	stack.push_back(std::make_pair(i, j));
	visit.at<int>(i, j) = stamp;
	while (!stack.empty())
	{
		std::pair<int, int> s;
		s = stack.back();
		stack.pop_back();
		ret.push_back(s);
		for (int k = 0; k < 4; k++)
		{
			int tx = s.first + dx[k];
			int ty = s.second + dy[k];
			if (tx >= 0 && tx < img.rows &&
					ty >= 0 && ty < img.cols &&
					visit.at<int>(tx, ty) < stamp &&
					std::abs(img.at<float>(tx, ty) - orig) < thr)
			{
				stack.push_back(std::make_pair(tx, ty));
				visit.at<int>(tx, ty) = stamp;
			}
		}
	}
	return ret;
}

void allComponents(const cv::Mat &img, std::vector<std::vector<std::pair<int, int> > > &components, std::vector<Box<int> > &boundingboxes, int sizeth, double blackth, double blackthu)
{
	std::vector<std::pair<int, int> > comp;
	Box<int> box;
	cv::Mat visit = cv::Mat::zeros(img.rows, img.cols, CV_32SC1);
	components.clear();
	boundingboxes.clear();
	int stamp = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (img.at<float>(i, j) >= 0.9 && //is black in original picture
					visit.at<int>(i, j) == 0)
			{
				comp = floodfill(img, i, j, visit, ++stamp);
				box = boundingBox(comp);
				box.maxx++; //upper bound exclusive
				box.maxy++;
				if (box.rangeX() < sizeth && box.rangeY() < sizeth)
					continue;
				double b = comp.size() / (double)box.area();
				if (b < blackth || b > blackthu)
					continue;
				components.push_back(comp);
				boundingboxes.push_back(box);
			}
}

std::vector<int> bestSplit(std::vector<double> hp, int minstep, int maxstep)
{
	int n = hp.size();
	//find best split using DP
	const int inf = ~0U >> 2;
	std::vector<int> split;
	std::vector<double> cost(hp.size() + 1);
	std::vector<int> decision(hp.size() + 1, -1);
	for (int i = 0; i < minstep; i++)
		cost[i] = inf;
	for (int j = minstep; j <= maxstep; j++)
		if (j < n)
			cost[j] = hp[j];
	hp.push_back(0);
	for (int i = maxstep + 1; i <= n; i++)
	{
		cost[i] = inf;
		decision[i] = -1;
		for (int j = minstep; j <= maxstep; j++)
			if (cost[i] > cost[i - j] + hp[i])
			{
				cost[i] = cost[i - j] + hp[i];
				decision[i] = i - j;
			}
	}
	int t = decision[n];
	while (t != -1)
	{
		split.push_back(t);
		t = decision[t];
	}
	return split;
}

std::vector<std::pair<int, int> > bestMerge(const cv::Mat &img, std::vector<std::vector<std::pair<int, int> > > &components, std::vector<Box<int> > &boxes, double tsx, double tsy, double blankcost, double splitsegcost, double hspthl, double hspth, std::vector<int> &rank)
{
	std::vector<std::pair<Box<int>, int> > boxind;
	for (int i = 0; i < boxes.size(); i++)
		boxind.push_back(std::make_pair(boxes[i], i));
	//split horizontally
	Box<int> box(0, img.rows, 0, img.cols);
	std::vector<std::vector<int> > hp(box.rangeX());
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (img.at<float>(i, j) >= 0.9) //is black originally
				hp[i - box.minx].push_back(j);
	std::vector<double> hpc(hp.size());
	for (int i = 0; i < hp.size(); i++)
	{
		std::sort(hp[i].begin(), hp[i].end());
		hpc[i] = hp[i].size() + blankcost;
		for (int j = 1; j < hp[i].size(); j++)
			if (hp[i][j - 1] != hp[i][j] - 1)
				hpc[i] += splitsegcost;
	}
	std::vector<int> hs = bestSplit(hpc, hspthl * tsx, hspth * tsx);
	std::sort(hs.begin(), hs.end());
	if (hs.back() != img.rows)
		hs.push_back(img.rows);
	//put components (boxes) in lines
	std::vector<std::vector<std::pair<Box<int>, int> > > boxindbuck(hs.size());
	for (std::pair<Box<int>, int> bi : boxind)
	{
		int c = bi.first.minx + bi.first.rangeX() / 2;
		int b;
		for (b = 0; b < hs.size() && c > hs[b]; b++);
		boxindbuck[b].push_back(bi);
		//fprintf(stderr, "put box %d in %d/%d\n", c, b, (int)hs.size());
	}
	std::vector<std::pair<int, int> > merge;
	rank.resize(boxind.size());
	int rankcnt = 0;
	for (std::vector<std::pair<Box<int>, int> > line : boxindbuck)
	{
		std::vector<std::pair<int, int> > tm = bestMergeLine(line, tsy);
		for (std::pair<Box<int>, int> x : line)
			rank[x.second] = rankcnt++;
		for (auto x : tm)
			merge.push_back(x);
	}
	return merge;
}

std::vector<std::pair<int, int> > bestMergeLine(std::vector<std::pair<Box<int>, int> > &boxind, double tsy)
{
	auto cost = [tsy](double x)
	{
		/*
		double d1 = tsy * 0.2, d2 = tsy * 1.2, inf = tsy * 100;
		double k = std::max(std::abs(x) - d1, 0.0);
		return k > d2 - d1 ? inf : k;
		*/
		return x * x;
	};
	auto cmp_miny = [](std::pair<Box<int>, int> a, std::pair<Box<int>, int> b)
	{
		return a.first.miny < b.first.miny;
	};
	std::sort(boxind.begin(), boxind.end(), cmp_miny);
	/*
	for (auto bi : boxind)
		fprintf(stderr, "box.miny = %d\n", bi.first.miny);
		*/
	std::vector<double> f(boxind.size() + 1);
	std::vector<int> prev(boxind.size() + 1);
	f[0] = 0;
	prev[0] = -1;
	for (int i = 1; i < f.size(); i++)
	{
		Box<int> c = boxind[i - 1].first;
		f[i] = f[i - 1] + cost(c.rangeY() - tsy);
		prev[i] = i - 1;
		for (int j = i - 2; j >= 0; j--)
		{
			c = uni(c, boxind[j].first);
			double t = f[j] + cost(c.rangeY() - tsy);
			if (f[i] > t)
			{
				f[i] = t;
				prev[i] = j;
			}
		}
		//fprintf(stderr, "f[%d] = %lf, prev = %d\n", i, f[i], prev[i]);
	}
	int t = boxind.size();
	std::vector<std::pair<int, int> > ret;
	while (prev[t] != -1)
	{
		for (int i = prev[t]; i < t; i++)
			ret.push_back(std::make_pair(boxind[prev[t]].second, boxind[i].second));
		t = prev[t];
	}
	return ret;
}

cv::Mat toMat(std::vector<float> v, Box<int> box)
{
	cv::Mat img = cv::Mat::zeros(box.rangeX(), box.rangeY(), CV_32FC1);
	int k = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			img.at<float>(i, j) = v[k++];
	return img;
}

cv::Mat toMat(std::vector<std::pair<int, int> > &points, Box<int> box)
{
	cv::Mat img = cv::Mat::zeros(box.rangeX(), box.rangeY(), CV_32FC1);
	for (std::pair<int, int> p : points)
		img.at<float>(p.first - box.minx, p.second - box.miny) = 1;
	return img;
}

Matrix toMatrix(std::vector<std::pair<int, int> > &points, Box<int> box)
{
	Matrix img(box.rangeX(), box.rangeY());
	for (std::pair<int, int> p : points)
		img.at(p.first - box.minx, p.second - box.miny) = 1;
	return img;
}

void toMat(std::vector<std::pair<int, int> > points, Box<int> box, cv::Mat &img)
{
	img = cv::Mat::zeros(box.rangeX(), box.rangeY(), CV_32FC1);
	for (std::pair<int, int> p : points)
		img.at<float>(p.first - box.minx, p.second - box.miny) = 1;
}

std::vector<float> toStdVector(const Matrix &m)
{
	return m.data;
}

std::vector<float> toStdVector(const cv::Mat &m)
{
	std::vector<float> vec;
	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++)
		{
			vec.push_back(m.at<float>(i, j));
		}
	return vec;
}
