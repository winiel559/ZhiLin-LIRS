#include <math.h>
#include <time.h>
#include <algorithm>
#include <set>
#include <assert.h>
#include "zlib/zlib.h"
#include "block.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

int main( int argc, char *argv[] )
{
	
	time_t startload_t;
	double init_load_t=0;
	
	
	//sleep(3);
	FILE *fp;
	fp = fopen(argv[1], "r");	
	
    struct problem total_prob;
	
	total_prob.bias = -1;
	
	printf("Input dir:%s\n", argv[1]);	
	
	//webspam
	if(strstr(argv[1], "webspam.train")!=0)
	{
		printf("Input training data: webspam.train\n");
		total_prob.l = 200000;
		total_prob.n = 16609143;
	}
	
	//epsilon
	else if(strstr(argv[1], "epsilon_normalized")!=0)
	{
		printf("Input training data: epsilon_normalized\n");
		total_prob.l = 400000;
		total_prob.n = 2000;
	}
	
	//kdd
	else if(strstr(argv[1], "kddb")!=0)
	{
		printf("Input training data: kddb\n");
		total_prob.l = 19264097;
		total_prob.n = 29890095;
	}
	
	//higgs
	else if(strstr(argv[1], "higgs")!=0)
	{
		printf("Input training data: higgs\n");
		total_prob.l = 10500000;
		total_prob.n = 28;
	}
	
	else
	{
		printf("\nUknown input file.\nUsage: ./preprocess_time training_file_name\n");
		printf("\nExample: \n");
		printf("./preprocess_time /media/disk/workload/SVM/webspam.train\n");
		printf("./preprocess_time /media/disk/workload/SVM/epsilon_normalized\n");
		printf("./preprocess_time /media/disk/workload/SVM/kddb\n");
		printf("./preprocess_time /media/disk/workload/SVM/higgs\n\n");
		
		return(0);
	}

	printf("l: %d\nn: %d\n", total_prob.l, total_prob.n);
	
	if ((strstr(argv[1], "epsilon_normalized")!=0)||(strstr(argv[1], "higgs")!=0))
	{
		printf("initial parsing & write time: 0\n\n");
		return 0;
	}
	
	
	char *buf = Malloc(char,1000000);
	startload_t = time(NULL);
	char  label_buf[2];
	
	for(int i=0; i<total_prob.l; i++)
	{
		//fseek(fp, offset[i], SEEK_SET);
		
		fgets(buf,1000000,fp);
		
		for(int j=0; j<2; j++)
			label_buf[j] = buf[j];
		
		int start = 3;
		
		char  index_buf[20] = "";
		char  value_buf[20] = "";
		
		for(int j=3; j<1000000; j++)
		{
			if(buf[j] == ':')
			{
				memcpy(index_buf,buf+start,j-start);
				start = j + 1 ; //value start
			}
			
			if(buf[j] == ' ')
			{
				memcpy(value_buf,buf+start,j-start);
				start = j + 1 ;
			}
			
			if(buf[j] == '\n' || buf[j] == '\0')
			{
				memcpy(value_buf,buf+start,j-start);
				break;
			}
			//printf("index_buf: %s, value_buf: %s\n", index_buf, value_buf);
		}
	}
	
	init_load_t += difftime(time(NULL), startload_t);
	//for(int i=0; i<200000; i++)
	//	printf("offset[%d] = %llu\n",i,offset[i]);

	printf("initial parsing & write time: %f\n\n",init_load_t);
	
}

