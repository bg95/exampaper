#pragma once

#include <string>
#include <cstdio>
#include <vector>
#include <unordered_map>
//#include "StringHash.h"

struct StringVectorHash
{
	std::hash<std::string> strhash;
	size_t operator()(std::vector<std::string> sv) const
	{
		size_t h = 0;
		for (std::string s : sv)
			h += strhash(s);
		return h;
	}
};
typedef std::unordered_map<std::string, int> StringIntMap;
typedef std::unordered_multimap<std::vector<std::string>, int, StringVectorHash> StringVectorIntMap;

struct CSVData
{
	std::vector<std::string> val;
	void read(FILE *fp);
	void write(FILE *fp) const;
	unsigned cols() const
	{
		return val.size();
	}
	const std::string &operator[](int s) const
	{
		return val[s];
	}
	std::string &operator[](int s)
	{
		return val[s];
	}
};

struct CSVHeader : public CSVData
{
	void read(FILE *fp); //also builds hash
	int getIndex(std::string col)
	{
		StringIntMap::iterator iter = hash.find(col);
		if (iter == hash.end())
			return -1;
		return (*iter).second;
	}
	void buildHash();
	StringIntMap hash;
};

struct CSVFile
{
	void read(FILE *fp)
	{
		read(fp, std::vector<std::string>());
	}
	void read(FILE *fp, std::string idtag)
	{
		read(fp, std::vector<std::string>(1, idtag));
	}
	void read(FILE *fp, std::vector<std::string> idtags);
	void write(FILE *fp) const
	{
		header.write(fp);
		for (CSVData d : data)
			d.write(fp);
	}
	int headerIndex(std::string col)
	{
		return header.getIndex(col);
	}
	int idIndex(std::vector<std::string> ids)
	{
		StringVectorIntMap::iterator iter = idhash.find(ids);
		if (iter == idhash.end())
			return -1;
		return (*iter).second;
	}
	std::vector<int> idIndices(std::string id)
	{
		return idIndices(std::vector<std::string>(1, id));
	}
	std::vector<int> idIndices(std::vector<std::string> ids)
	{
		auto range = idhash.equal_range(ids);
		std::vector<int> r;
		for (auto iter = range.first; iter != range.second; iter++)
			r.push_back((*iter).second);
		return r;
	}
	int idIndex(std::string id)
	{
		return idIndex(std::vector<std::string>(1, id));
	}
	size_t rows() const
	{
		return data.size();
	}
	const CSVData &operator[](int s) const
	{
		return data[s];
	}
	CSVData &operator[](int s)
	{
		return data[s];
	}
	CSVHeader header;
	std::vector<CSVData> data;
	StringVectorIntMap idhash;
};

struct CSVStream
{
	void init(FILE *fp);
	int headerIndex(std::string col)
	{
		return header.getIndex(col);
	}
	CSVData getNext();
	bool eof();

	bool next();
	CSVHeader header;
	CSVData td;
	FILE *fp;
	bool ateof;
};
