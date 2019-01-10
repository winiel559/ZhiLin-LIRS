#include <math.h>
#include <time.h>
#include <algorithm>
#include <set>
#include <assert.h>
#include "zlib/zlib.h"
#include "block.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define nBlks 40

void int2str(int , char *);

int main( int argc, char *argv[] )
{
	
	time_t startload_t;
	time_t startscan_t;
	time_t startwrite_t;
	double init_load_t=0;
	double scan_t=0;
	double write_t=0;
	
	
	
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
		printf("\nUknown input file.\nUsage: ./splitting_time /media/3DXpoint/workload/SVM/kddb .40/\n");
		
		return(0);
	}

	printf("l: %d\nn: %d\n", total_prob.l, total_prob.n);
	
	
	char dir[1000]; //= "/media/3DXpoint/workload/SVM/kddb/kddb.40/kddb_";
	char split_dir[nBlks][1000];
	char file_id[10];
	
	strcpy(dir, argv[1]);
	strcat(dir, argv[2]);
	//printf("%s\n",dir);
	
	for(int i=0; i<nBlks; i++)
	{
		int2str(i, file_id);
		strcpy(split_dir[i], dir);
		strcat(split_dir[i], "higgs_");
		strcat(split_dir[i], file_id);
		strcat(split_dir[i], ".sep");
		// printf("%s\n",split_dir[i]);
	}
	
	FILE *write_file[nBlks];
	
	for(int i=0; i<nBlks; i++)
	{
		write_file[i] = fopen(split_dir[i],"wb+");
		//fwrite(&offset[0], sizeof(offset[0]), 1, write_file[i]);
	}

	char *buf = Malloc(char,1000000);
	char  label_buf[2];
	int i = 0;
	int instance_size = 0;
	startload_t = time(NULL);
	
	while(fgets(buf,1000000,fp) != NULL)
    {
		
		startscan_t = time(NULL);
		instance_size = 0;

		
		for(int j=0; j<2; j++)
		{
			label_buf[j] = buf[j];
			instance_size += 1;
		}
			
		int start = 3;
		
		char  index_buf[2000] = "";
		char  value_buf[2000] = "";
		
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
			
			instance_size += 1;
			//printf("index_buf: %s, value_buf: %s\n", index_buf, value_buf);
		}
		
		scan_t += difftime(time(NULL), startscan_t);
		
		startwrite_t = time(NULL);
		
		for(int j=0; j<instance_size; j++)
		{
			fwrite(&buf, sizeof(char), 1, write_file[i%nBlks]);	
		}
		fflush(write_file[i%nBlks]);
		write_t += difftime(time(NULL), startwrite_t);
		
		/*
		fwrite(&buf, instance_size, 1, write_file[i%40]);
		fflush(write_file[i%40]);
		*/
		i++;
	}
	
	
	for(int i=0; i<nBlks; i++)
	{
		fclose(write_file[i]);
	}
	
	
	
	
	init_load_t += difftime(time(NULL), startload_t);
	//for(int i=0; i<200000; i++)
	//	printf("offset[%d] = %llu\n",i,offset[i]);

	// printf("initial splitting time: %f scan_time: %f write_time: %f\n",init_load_t, scan_t, write_t);
	printf("initial splitting time: %f \n\n",init_load_t);
	
}

void int2str(int i, char *s) 
{
  sprintf(s,"%d",i);
}

