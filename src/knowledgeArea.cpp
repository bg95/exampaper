#include "CSVData.h"

int main()
{
	const std::string dir("../data/");
	CSVFile questionknowledge, studentarea, knowledge, question;
	{
		FILE *qk, *sa, *k, *q;
		qk = fopen((dir + "tifen_questionknowledge.csv").data(), "r");
		sa = fopen((dir + "student.csv").data(), "r");
		q = fopen((dir + "tifen_question.csv").data(), "r");
		k = fopen((dir + "tifen_knowledge.csv").data(), "r");
		questionknowledge.read(qk, "QUESTION_ID"); //QUESTION_ID -> KNOWLEDGE_ID, should be one to multiple
		studentarea.read(sa, "学生ID"); //STUDENT_ID -> CLASS_ID
		knowledge.read(k, "CODE"); //KNOWLEDGE_ID -> KNOWLEDGE_NAME
		question.read(q, std::vector<std::string>({"PAPER_ID", "QUESTION_ORDER"})); //PAPER_ID, QUESTION_ORDER -> QUESTION_ID
		fclose(qk);
		fclose(sa);
		fclose(k);
		fclose(q);
	}
	int qidcol = question.headerIndex("ID");
	int qscorecol = question.headerIndex("SCORES");
	int kidcol = questionknowledge.headerIndex("KNOWLEDGE_ID");
	int kncol = knowledge.headerIndex("NAME");
	int kkidcol = knowledge.headerIndex("CODE");
	int classidcol = studentarea.headerIndex("班级ID");
	CSVStream studentquestion;
	FILE *sq;
	sq = fopen((dir + "tifen_studentquestion.csv").data(), "r");
	studentquestion.init(sq); //STUDENT_ID -> PAPER_ID, QUESTION_ORDER
	int studentcol = studentquestion.headerIndex("USER");
	int scorecol = studentquestion.headerIndex("SCORES");
	int paperidcol = studentquestion.headerIndex("PAPER_ID");
	int questionordercol = studentquestion.headerIndex("QUESTION_ORDER");
	std::unordered_map<std::string, std::unordered_map<std::string, std::pair<int, int> > > skscore;
	while (!studentquestion.eof())
	{
		CSVData td = studentquestion.getNext();
		std::string student = td[studentcol];
		std::string score = td[scorecol];

		int si = studentarea.idIndex(student);
		if (si == -1)
		{
			fprintf(stderr, "Warning: student %s not found!\n", student.data());
			continue;
		}
		std::string classid = studentarea[si][classidcol];

		std::vector<std::string> pidqor;
		pidqor.push_back(td[paperidcol]);
		pidqor.push_back(td[questionordercol]);
		int qi = question.idIndex(pidqor);
		if (qi == -1)
		{
			fprintf(stderr, "Warning: paper %s, question %s not found!\n", pidqor[0].data(), pidqor[1].data());
			continue;
		}
		std::string qid = question[qi][qidcol];
		std::string qscore = question[qi][qscorecol];

		std::vector<int> kis = questionknowledge.idIndices(qid);
		//printf("student %s, class %s, question %s, knowledge", student.data(), classid.data(), qid.data());
		for (int ki : kis)
		{
			std::string kid = questionknowledge[ki][kidcol];
			skscore[student][kid].first += atoi(score.data());
			skscore[student][kid].second += atoi(qscore.data());
			int kni = knowledge.idIndex(kid);
			if (kni == -1)
			{
				fprintf(stderr, "Warning: knowledge %s not found!\n", kid.data());
				continue;
			}
			std::string kn = knowledge[kni][kncol];
			//printf(" %s", kn.data());
		}
		//printf(", score %s/%s\n", score.data(), qscore.data());

		/*
		int ki = questionknowledge.idIndex(qid);
		if (ki == -1)
		{
			fprintf(stderr, "Warning: question %s not found!\n", qid.data());
			continue;
		}
		std::string kid = questionknowledge[ki][kidcol];
		int kni = knowledge.idIndex(kid);
		if (kni == -1)
		{
			fprintf(stderr, "Warning: knowledge %s not found!\n", kid.data());
			continue;
		}
		std::string kn = knowledge[kni][kncol];
		printf("student %s, class %s, question %s, knowledge %s, score %s/%s\n",
				student.data(), classid.data(), qid.data(), kn.data(), score.data(), qscore.data());
		*/
	}
	/*
	for (auto siter : skscore)
	{
		printf("student %s\n", siter.first.data());
		for (auto kiter : siter.second)
			printf("\tknowledge %s: %d/%d = %.2lf\n", (kiter).first.data(), (kiter).second.first, (kiter).second.second, (double)(kiter).second.first / (kiter).second.second);
	}
	*/
	const int buckets = 10;
	std::unordered_map<std::string, std::vector<double> > acqdis; //knowledge acquirement distribution
	for (auto siter : skscore)
	{
		for (auto kiter : siter.second)
		{
			acqdis[kiter.first].resize(buckets + 1);
			acqdis[kiter.first][buckets * (kiter).second.first / (kiter).second.second] += 1.0;
		}
	}
	for (auto &disiter : acqdis) //normalize
	{
		double s = 0;
		for (double x : disiter.second)
			s += x;
		for (double &x : disiter.second)
			x /= s;
	}
	/*
	for (CSVData k : knowledge.data)
	{
		std::string kid = k[kkidcol];
		auto iter = acqdis.find(kid);
		if (iter != acqdis.end())
		{
			printf("knowledge %s (%s):", k[kncol].data(), kid.data());
			for (auto x : (*iter).second)
				printf(" %.2lf", x);
			printf("\n");
		}
	}
	*/
	std::unordered_map<std::string, std::vector<std::pair<double, int> > > akscore;
	for (auto siter : skscore)
	{
		int ai = studentarea.idIndex(siter.first);
		if (ai == -1)
		{
			fprintf(stderr, "Warning: student %s not found!\n", siter.first.data());
			continue;
		}
		std::string area = studentarea[ai][classidcol];
		for (auto kiter : siter.second)
		{
			akscore[area].resize(knowledge.rows());
			std::pair<double, int> &ent = akscore[area][knowledge.idIndex(kiter.first)];
			ent.first += (double)(kiter).second.first / (kiter).second.second;
			ent.second++;
		}
	}
	for (int i = 0; i < knowledge.rows(); i++)
		printf("\t%s", knowledge[i][kncol].data());
	printf("\n");
	for (auto aiter : akscore)
	{
		printf("%s", aiter.first.data()); //classid
		for (int i = 0; i < knowledge.rows(); i++)
			printf("\t%.2lf(%d)", aiter.second[i].first / aiter.second[i].second, aiter.second[i].second);
		printf("\n");
	}
	return 0;
}
