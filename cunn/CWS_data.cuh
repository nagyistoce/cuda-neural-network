#pragma once 
#include "..\\..\\..\\wenx.h"
#include <map>
#include <string>
namespace CWS{
	const int k_number_of_input=5;
		//a window of size 5
	const int k_number_of_output=4;
		//4 tag as B,I,E,S
	const int k_max_number_of_data=1000000;
		//get 1M windows at most
	const int k_max_number_of_training=800000;
	const int k_max_number_of_testing=200000;
	const int k_dict_size=5000;
}

using namespace std;
typedef int DataIn[CWS::k_max_number_of_data][CWS::k_number_of_input];
typedef int DataOut[CWS::k_max_number_of_data][CWS::k_number_of_output];
class CWS_data{
public:
	map<wchar_t,int> DictChar;
	DataIn  *data_in;
	DataOut *data_out;
	int dictsize;

	CWS_data(string Dictfilename,string Infilename,
		string Tagfilename){

		//input the Dict
		FILE *DICT=fopen(Dictfilename.c_str(),"r");
		pair<wchar_t,int> pr;
		pr.first=L' ';pr.second=0;DictChar.insert(pr);
		//Space means not found in the dictionary..
		pr.first=L'¡¾';pr.second=1;DictChar.insert(pr);
		pr.first=L'¡¿';pr.second=2;DictChar.insert(pr);
		//¡¾ is used for begining and 
		//¡¿ is used for ending
		wchar_t chr;
		wchar_t tline[1000];
		for(int i=3;i<CWS::k_dict_size;i++){
			if(!fgetws(tline,1000,DICT))
				break;//???? is it right?
			chr=tline[0];
			if(DictChar.find(chr)!=DictChar.end())
				err("Multiple same char",__FILE__,__LINE__);
			pr.first=chr;pr.second=i;
			DictChar.insert(pr);
		}

		FILE* IN=fopen(Infilename.c_str(),"r");
		FILE* TAGIN=fopen(Tagfilename.c_str(),"r");
		wchar_t s[5000];
		char stag[5000];
		s[0]=s[1]=L'¡¾';
		int slen;
		int sct=0;
		int data_counter=0;
		data_in=(DataIn*)malloc(sizeof(DataIn));
		data_out=(DataOut*)malloc(sizeof(DataIn));
		if(data_in==NULL || data_out==NULL)
			err("malloc failed",__FILE__,__LINE__);

		memset(data_out,0,sizeof(DataOut));

		while(fgetws(s+2,4990,IN)){
			fgets(stag,5000,TAGIN);
			if(++sct%1000==0)
				printf("Finish reading %d sentences, %d chars\n",sct,data_counter);
			slen=wcslen(s)-1;//in case of '\n'
			s[slen]=L'¡¿';
			s[slen+1]=L'¡¿';
			s[slen+2]=0;
			for(int j=2;j<slen;j++){
				(*data_in)[data_counter][0]=DictChar[s[j-2]];
				(*data_in)[data_counter][1]=DictChar[s[j-1]];
				(*data_in)[data_counter][2]=DictChar[s[j-0]];
				(*data_in)[data_counter][3]=DictChar[s[j+1]];
				(*data_in)[data_counter][4]=DictChar[s[j+2]];

				switch (stag[j-2])
				{
					//All elements in data_out hase been set Zero at
					//memset(data_out,0,sizeof(DataOut));
				case 'B':(*data_out)[data_counter][0]=1;break;
				case 'I':(*data_out)[data_counter][1]=1;break;
				case 'E':(*data_out)[data_counter][2]=1;break;
				case 'S':(*data_out)[data_counter][3]=1;break;
				default:
					err("Illegal Tags...",__FILE__,__LINE__);
					break;
				}
				data_counter++;
				if(data_counter>=CWS::k_max_number_of_data)
					break;
			}// for j (each word in a sentences)
			if(data_counter>=CWS::k_max_number_of_data)
					break;
		}
		if(data_counter>=CWS::k_max_number_of_data)
			printf("Loading successfully.\n");
		else
			err("NOT enough data",__FILE__,__LINE__);
	}
};
