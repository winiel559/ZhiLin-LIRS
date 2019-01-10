#include <math.h>
#include <time.h>
#include <algorithm>
#include <set>
#include <assert.h>
#include "block.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
int main( int argc, char *argv[] )
{
	time_t startload_t;
	double init_load_t=0;
	
	startload_t = time(NULL);
	//sleep(3);
	FILE *fp;
	fp = fopen(argv[1], "r");	
	//fp = fopen("../data/webspam.train", "r");
	//fp = fopen("123", "r");
	
    struct problem total_prob;
	
	total_prob.bias = -1;
	
	/*debug
	printf("test: %d\n", strstr(argv[1], "webspam.train"));
	printf("test: %d\n", strstr(argv[1], "epsilon_normalized"));
	printf("test: %d\n", strstr(argv[1], "kddb"));
	*/
	
	//webspam
	if(strstr(argv[1], "webspam.train")!=0)
	{
		total_prob.l = 200000;
		total_prob.n = 16609143;
	}
	
	//epsilon
	else if(strstr(argv[1], "epsilon_normalized")!=0)
	{
		total_prob.l = 400000;
		total_prob.n = 2000;
	}
	
	//kdd
	else if(strstr(argv[1], "kddb")!=0)
	{
		total_prob.l = 19264097;
		total_prob.n = 29890095;
	}
	
	//higgs
	else if(strstr(argv[1], "higgs")!=0)
	{
		total_prob.l = 10500000;
		total_prob.n = 28;
	}
	
	/*debug
	printf("test: %d %d\n", total_prob.l, total_prob.n);
	*/
	unsigned long long *offset = Malloc(unsigned long long,total_prob.l+1);
	char *buf = Malloc(char,1000000);
	int i = 0;
	unsigned long long accu = 0;
	
	while(fgets(buf,1000000,fp) != NULL)
    {
		for(int j=0; j<1000000; j++)
			if(buf[j] == '\n' || buf[j] == '\0')
			{
				offset[i] = accu;
				accu += j+1;
				if(i%(total_prob.l/10)==0)
					printf("offset[%d]: %llu\n", i, offset[i]);
				break;
			}	
		i++;
	}
	
	//int num_feature[total_prob.l];
	int *num_feature = Malloc(int,total_prob.l);
	
	feature_node **node = Malloc(feature_node*,total_prob.l);
	int *y = Malloc(int,total_prob.l);
	
	total_prob.y = y;
	feature_node *node_buffer = Malloc(feature_node,total_prob.n);
	
	char  label_buf[2];
	int max_feature = 0;
	
	for(int i=0; i<total_prob.l; i++)
	{
		if(i%(total_prob.l/10)==0)
			printf("%d/%d\n", i, total_prob.l);
		
		fseek(fp, offset[i], SEEK_SET);
		
		fgets(buf,1000000,fp);
		
		//for(int j=0; j<100; j++)
			//printf("%c",buf[j]);
		
		for(int j=0; j<2; j++)
			label_buf[j] = buf[j];
		
		total_prob.y[i] = atoi(label_buf);
		// printf("y=%d\n", total_prob.y[i]);
		
		int start;
		if(total_prob.y[i]==-1)
			start = 3;
		else
			start = 2;
		
		feature_node *xi = node_buffer;
		
		
		num_feature[i]=0;
		
		for(int j=start; j<1000000; j++)
		{
			char  index_buf[50] = "";
			char  value_buf[50] = "";
			if(buf[j] == ':')
			{
				/*
				for(int c=start; c<j; c++)
				{
					index_buf[c-start] = buf[c];
				}
				*/
				memcpy(index_buf,buf+start,j-start);
				xi->index = atoi(index_buf);
				start = j + 1 ; //value start
			}
			
			if(buf[j] == ' ')
			{
				/*
				for(int c=start; c<j; c++)
				{
					value_buf[c-start] = buf[c];
				}
				*/
				memcpy(value_buf,buf+start,j-start);
				xi->value = atof(value_buf);
				// if(i<10000)
					// printf("xi->value: %f\n", xi->value);
				num_feature[i]++;
				xi++;
				start = j + 1 ;
			}
			
			if(buf[j] == '\n' || buf[j] == '\0')
			{
				/*
				for(int c=start; c<j; c++)
				{
					value_buf[c-start] = buf[c];
				}
				*/
				memcpy(value_buf,buf+start,j-start);
				xi->value = atof(value_buf);
				// printf("xi->value: %f\n", xi->value);
				num_feature[i]++;
				xi++;
				break;
			}
			//xi--;
			//printf("xi->index: %d, xi->value: %f\n", xi->index, xi->value);
			//xi++;
		}
		
		//if(i%10000==0)
			//printf("num_feature[%d]: %d\n", i, num_feature[i]);
		
		//Find the max feature number
		// if(num_feature[i] > max_feature)
		// {
			// max_feature = num_feature[i];
			// printf("num_feature[%d]: %d, max_feature: %d\n", i, num_feature[i], max_feature);
		// }
			
		node[i] = Malloc(feature_node,num_feature[i]+1);
		for(int j=0; j<num_feature[i]+1; j++)
			node[i][j] = node_buffer[j];
		
		total_prob.x = node;	
		
	}
	
	FILE *write_file;
	write_file = fopen(argv[2],"wb+");
	
	
	fwrite(&num_feature[0], sizeof(num_feature[0]), total_prob.l, write_file);
	
	/*
	for(int i=0; i<10000; i++)
	{
		for(int j=0; j<num_feature[i]; j++)
		{
			printf("index[i][j]:%d, value[i][j]:%f\n",total_prob.x[i][j].index, total_prob.x[i][j].value);
		}
	}
	*/
	
	for(int i=0; i<total_prob.l; i++)
	{
		fwrite(&total_prob.y[i], sizeof(total_prob.y[i]), 1, write_file);
		for(int j=0; j<num_feature[i]; j++)
			fwrite(&total_prob.x[i][j].index, sizeof(total_prob.x[i][j].index), 1, write_file);
		for(int j=0; j<num_feature[i]; j++)
			fwrite(&total_prob.x[i][j].value, sizeof(total_prob.x[i][j].value), 1, write_file);
	}
	
	fclose(write_file);
//---------------------end
	/*
	int label_buf2[10];
	int index_buf[10];
	FILE *read_binary;
	read_binary = fopen("binary_file","rb"); 
	unsigned long long *byte_offset = Malloc(unsigned long long,total_prob.l+1);
	byte_offset[0] = total_prob.l * 4;
	for(int i=1; i<total_prob.l; i++)
	{
		byte_offset[i] = byte_offset[i-1] + 4 + 12 * num_feature[i-1];
	}
	
	for(int i=0; i<10; i++)
	{
		fseek(read_binary, byte_offset[i], SEEK_SET);
		fread(&label_buf2[i], sizeof(int), 1, read_binary);
		printf("label[%d]: %d\n", i, label_buf2[i]);
		
		fseek(read_binary, byte_offset[i]+4, SEEK_SET);
		fread(&index_buf[0], sizeof(int), 10, read_binary);
		for(int j=0; j<10; j++)
		{
			printf("index_buf[%d]: %d\n", j, index_buf[j]);
		}	
	}
	*/
	init_load_t += difftime(time(NULL), startload_t);
	//for(int i=0; i<200000; i++)
		//printf("offset[%d] = %llu\n",i,offset[i]);
	fclose(fp);
	printf("initial parsing & write time: %f\n",init_load_t);
	
}