#pragma once

#include <vector>
#include <string>
#include <cstring>

template <class T>
struct StringHashLink
{
	std::string s;
	T v;
	StringHashLink *next;
};

template <class T>
class StringHash
{
public:

	class Iterator
	{
	public:
		StringHash<T> *table;
		unsigned pos;
		StringHashLink<T> *h;
		bool at_end;
		Iterator(StringHash *table)
			:table(table)
		{
			pos = 0;
			h = 0;
			at_end = false;
			while (pos < table->size)
			{
				if (table->val[pos])
				{
					h = table->val[pos];
					return;
				}
				pos++;
			}
			at_end = true;
		}
		bool atEnd()
		{
			return at_end;
		}
		void operator++(int)
		{
			if (h && h->next)
			{
				h = h->next;
				return;
			}
			pos++;
			while (pos < table->size)
			{
				if (table->val[pos])
				{
					h = table->val[pos];
					return;
				}
				pos++;
			}
			at_end = true;
		}
		std::string string()
		{
			return h->s;
		}
		T &value()
		{
			return h->v;
		}
	};

	StringHash(unsigned size = 16385)
		:size(size)
	{
		val = new StringHashLink<T> *[size];
		memset(val, 0, size * sizeof(val[0]));
		num_keys = 0;
	}
	~StringHash()
	{
		clear();
		delete val;
	}
	Iterator begin()
	{
		return Iterator(this);
	}
	bool insert(std::string s, T v)
	{
		unsigned pos = hash(s.data());
		StringHashLink<T> *h = val[pos];
		while (h)
		{
			if (h->s == s)
			{
				h->v = v;
				return false;
			}
			h = h->next;
		}
		h = new StringHashLink<T>;
		h->s = s;
		h->v = v;
		h->next = val[pos];
		val[pos] = h;
		num_keys++;
		checkResize();
		return true;
	}
	bool insertWithoutReplace(std::string s, T v)
	{
		unsigned pos = hash(s.data());
		StringHashLink<T> *h = val[pos];
		while (h)
		{
			if (h->s == s)
			{
				//no replace
				return false;
			}
			h = h->next;
		}
		h = new StringHashLink<T>;
		h->s = s;
		h->v = v;
		h->next = val[pos];
		val[pos] = h;
		num_keys++;
		checkResize();
		return true;
	}
	void clear()
	{
		for (int i = 0; i < size; i++)
		{
			StringHashLink<T> *h = val[i], *g;
			while (h)
			{
				g = h->next;
				delete h;
				h = g;
			}
		}
		memset(val, 0, size * sizeof(val[0]));
		num_keys = 0;
	}
	unsigned numKeys()
	{
		return num_keys;
	}
	T &operator[](std::string s)
	{
		unsigned pos = hash(s.data());
		StringHashLink<T> *h = val[pos];
		while (h)
		{
			if (h->s == s)
			{
				return h->v;
			}
			h = h->next;
		}
		h = new StringHashLink<T>;
		h->s = s;
		h->v = T();
		h->next = val[pos];
		val[pos] = h;
		num_keys++;
		if (checkResize())
			return (*this)[s];
		return h->v;
	}

private:
	StringHashLink<T> **val;
	unsigned size, num_keys;
	unsigned hash(const char *s)
	{
		unsigned h = 0;
		while (*s)
		{
			h = h * 31 + *s;
			s++;
		}
		return h % size;
	}
	bool checkResize()
	{
		if (num_keys > size / 2)
		{
			resize(size * 2 - 1);
			return true;
		}
		return false;
	}
	void resize(unsigned new_size)
	{
		StringHashLink<T> **oldval = val;
		unsigned oldsize = size;
		size = new_size;
		val = new StringHashLink<T> *[size];
		memset(val, 0, size * sizeof(val[0]));
		num_keys = 0;
		for (int i = 0; i < oldsize; i++)
		{
			StringHashLink<T> *h = oldval[i], *g;
			while (h)
			{
				insert(h->s, h->v);
				g = h->next;
				delete h;
				h = g;
			}
		}
		delete oldval;
	}
};

