#include <cstring>
#include "CSVData.h"

void CSVData::read(FILE *fp)
{
	int c;
	std::string str;
	val.clear();
	do
	{
		str.clear();
		c = getc(fp);
		if (c == '{')
		{
			while (c != -1 && c != '}' && c != '\n')
			{
				str.push_back(c);
				c = getc(fp);
			}
			if (c == '}')
				str.push_back(c);
		}
		else
		{
			while (c != -1 && c != ',' && c != '\n')
			{
				str.push_back(c);
				c = getc(fp);
			}
		}
		val.push_back(str);
	} while (c != '\n' && c != -1);
	if (val.size() == 1 && val[0].size() == 0)
		val.clear();
}
void CSVData::write(FILE *fp) const
{
	int i;
	for (i = 0; i < cols() - 1; i++)
		fprintf(fp, "%s,", val[i].data());
	fprintf(fp, "%s\n", val[i].data());
}

void CSVHeader::read(FILE *fp)
{
	CSVData::read(fp);
	buildHash();
}
void CSVHeader::buildHash()
{
	hash.clear();
	for (int i = 0; i < cols(); i++)
		hash.insert(std::make_pair(val[i], i));
}

void CSVFile::read(FILE *fp, std::vector<std::string> idtags)
{
	header.read(fp);
	std::vector<int> idcols;
	for (std::string &tag : idtags)
	{
		int idcol = headerIndex(tag);
		if (idcol == -1)
			fprintf(stderr, "Error: column %s not found!\n", tag.data());
		else
			idcols.push_back(idcol);
	}
	data.clear();
	idhash.clear();
	CSVData td;
	while (!feof(fp))
	{
		td.read(fp);
		if (!td.cols())
			break;
		if (td.cols() != header.cols())
			fprintf(stderr, "Warning: number of columns does not match! (header %d, data %d)\n", (int)header.cols(), (int)td.cols());
		if (idcols.size())
		{
			std::vector<std::string> ids;
			for (int c : idcols)
				ids.push_back(td[c]);
			idhash.insert(std::make_pair(ids, data.size()));
			/*
			fprintf(stderr, "idhash: ");
			for (std::string c : ids)
				fprintf(stderr, " %s", c.data());
			fprintf(stderr, " -> %d\n", (int)data.size());
			auto iter = idhash[ids];
			fprintf(stderr, "find %d\n", (int)(iter));
			*/
		}
		data.push_back(td);
		if (data.size() % 100000 == 0)
			fprintf(stderr, "Info: %d rows read\n", (int)data.size());
	}
}

void CSVStream::init(FILE *fp)
{
	this->fp = fp;
	header.read(fp);
	ateof = next();
}
CSVData CSVStream::getNext()
{
	CSVData t = td;
	ateof = next();
	return t;
}
bool CSVStream::eof()
{
	return ateof;
}
bool CSVStream::next()
{
	if (!feof(fp))
	{
		td.read(fp);
		if (!td.cols())
			return true; //end of file
		if (td.cols() != header.cols())
			fprintf(stderr, "Warning: number of columns does not match! (header %d, data %d)\n", (int)header.cols(), (int)td.cols());
		return false;
	}
	return true;
}
